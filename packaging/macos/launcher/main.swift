// Prometheus.app's launcher: the app's main executable, one binary with several modes.
//
//   Prometheus --register      register the LaunchAgent with SMAppService (idempotent)
//   Prometheus --unregister    stop and unregister it (the way to swap the app for an update)
//   Prometheus --status        not_registered | registered | requires_approval | running
//   Prometheus --pair          write a fresh one-time pairing secret (re-pair after a Beacon reinstall)
//   Prometheus --uninstall [--purge-data]
//                              unregister, remove the pairing secret, move the app to the Trash;
//                              ~/.prometheus and the env file stay unless --purge-data
//   Prometheus --run           what launchd starts: set the environment, then exec the bundled Python
//   Prometheus                 (no arguments, a person double-clicked it) one small window
//
// Every mode except --run and the window prints ONE JSON line on stdout and exits with a distinct
// code, so the caller (Beacon) reads state instead of polling. The contract is in
// docs/design/macos-app-installer.md.
//
// Which agent plist to register comes from this bundle's Info.plist (PrometheusAgentPlist,
// PrometheusAgentLabel), not from constants here, so a development bundle with different identifiers
// runs the identical code.

import AppKit
import Foundation
import ServiceManagement

// MARK: - Contract

enum Code: Int32 {
    case ok = 0
    case usage = 2
    case requiresApproval = 10
    case alreadyRunning = 11
    case portBusy = 12
    case failed = 13
    case unsupportedOS = 14
}

/// What a mode concluded. The CLI prints it; the window shows it.
struct Outcome {
    var code: Code
    var state: String
    var detail: String = ""
    var extra: [String: Any] = [:]
    var ok: Bool { code == .ok }
}

/// The REST port the daemon serves on. Setup mode and the configured daemon both use it.
let daemonPort: UInt16 = 8005

let info = Bundle.main.infoDictionary ?? [:]
let agentLabel = info["PrometheusAgentLabel"] as? String ?? "com.oaralabs.prometheus.daemon"
let agentPlist = info["PrometheusAgentPlist"] as? String ?? "\(agentLabel).plist"
let appVersion = info["CFBundleShortVersionString"] as? String ?? "unknown"
let beaconBundleID = "com.oaralabs.beacon"
let beaconURL = URL(string: "https://oara.ai/beacon")!

/// Print the outcome as one JSON line and exit with its code.
func finish(_ outcome: Outcome) -> Never {
    var object: [String: Any] = [
        "ok": outcome.ok,
        "state": outcome.state,
        "detail": outcome.detail,
        "agent": agentLabel,
        "app_version": appVersion,
    ]
    for (key, value) in outcome.extra { object[key] = value }
    let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(outcome.code.rawValue)
}

// MARK: - Is something on the daemon's port, and is it a Prometheus?

enum PortState {
    case free
    case prometheus
    case other
}

/// One short HTTP probe of 127.0.0.1:<daemonPort>. A refused connection is "free". A Prometheus is
/// recognised the way a client can recognise one today: setup mode answers /api/setup/status with
/// {"setup_mode": true, ...}, and a configured daemon answers any /api/ path with 401 and a JSON error that
/// names both "unauthorized" and "Bearer".
func probeDaemonPort() -> PortState {
    func fetch(_ path: String) -> (status: Int, body: Data)? {
        guard let url = URL(string: "http://127.0.0.1:\(daemonPort)\(path)") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.5
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let semaphore = DispatchSemaphore(value: 0)
        var result: (Int, Data)?
        var refused = false
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let http = response as? HTTPURLResponse { result = (http.statusCode, data ?? Data()) }
            if let urlError = error as? URLError, urlError.code == .cannotConnectToHost { refused = true }
            semaphore.signal()
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + 3)
        if refused { return (-1, Data()) }
        return result
    }

    guard let first = fetch("/api/setup/status") else { return .other }
    if first.status == -1 { return .free }
    if first.status == 200,
       let json = try? JSONSerialization.jsonObject(with: first.body) as? [String: Any],
       (json["setup_mode"] as? Bool) == true {
        return .prometheus
    }
    // A configured daemon has no /api/setup/*: any /api/ path answers the bearer middleware's 401,
    // whose JSON error names both "unauthorized" and "Bearer". Beacon applies the same two-word test,
    // so that someone else's login wall on this port is not mistaken for Prometheus. Reword that error
    // only keeping both words (web/server.py, _check_bearer_token), and tell Beacon first.
    if let second = fetch("/api/status"), second.status == 401,
       let json = try? JSONSerialization.jsonObject(with: second.body) as? [String: Any],
       let error = json["error"] as? String {
        let text = error.lowercased()
        if (text.contains("unauthorized") || text.contains("unauthorised")) && text.contains("bearer") {
            return .prometheus
        }
    }
    return .other
}

// MARK: - Registration

let approvalHint = "Allow Prometheus under System Settings > General > Login Items."

@available(macOS 13.0, *)
func stateName(_ status: SMAppService.Status) -> String {
    switch status {
    case .notRegistered: return "not_registered"
    case .enabled: return "registered"
    case .requiresApproval: return "requires_approval"
    case .notFound: return "not_found"
    @unknown default: return "unknown"
    }
}

@available(macOS 13.0, *)
func describe(_ error: Error) -> String {
    let nsError = error as NSError
    return "\(nsError.localizedDescription) (\(nsError.domain) \(nsError.code))"
}

@available(macOS 13.0, *)
func needsApproval() -> Outcome {
    SMAppService.openSystemSettingsLoginItems()
    return Outcome(code: .requiresApproval, state: "requires_approval", detail: approvalHint)
}

@available(macOS 13.0, *)
func register() -> Outcome {
    let service = SMAppService.agent(plistName: agentPlist)

    switch service.status {
    case .enabled:
        return Outcome(code: .ok, state: "already_registered")
    case .requiresApproval:
        return needsApproval()
    case .notFound, .notRegistered:
        // `.notFound` is not trusted before a first attempt: it is what status reports until
        // register() has run, and register()'s own error says what is actually wrong.
        break
    @unknown default:
        break
    }

    // Not ours yet: refuse rather than start a second daemon next to someone else's.
    switch probeDaemonPort() {
    case .prometheus:
        return Outcome(code: .alreadyRunning, state: "already_running",
                       detail: "A Prometheus already answers on 127.0.0.1:\(daemonPort).")
    case .other:
        return Outcome(code: .portBusy, state: "port_busy",
                       detail: "Something that is not Prometheus holds 127.0.0.1:\(daemonPort).")
    case .free:
        break
    }

    do {
        try service.register()
    } catch {
        // register() throws for "already registered" and "needs approval" as well as real failures;
        // the status afterwards tells them apart better than the error code does.
        switch service.status {
        case .enabled: return Outcome(code: .ok, state: "already_registered")
        case .requiresApproval: return needsApproval()
        default: return Outcome(code: .failed, state: "registration_failed", detail: describe(error))
        }
    }

    switch service.status {
    case .enabled:
        return Outcome(code: .ok, state: "registered")
    case .requiresApproval:
        return needsApproval()
    default:
        return Outcome(code: .failed, state: "registration_failed",
                       detail: "register() returned but the status is \(stateName(service.status)).")
    }
}

@available(macOS 13.0, *)
func unregister() -> Outcome {
    let service = SMAppService.agent(plistName: agentPlist)
    if service.status == .notRegistered || service.status == .notFound {
        return Outcome(code: .ok, state: "not_registered")
    }
    do {
        try service.unregister()
    } catch {
        return Outcome(code: .failed, state: "registration_failed", detail: describe(error))
    }
    return Outcome(code: .ok, state: "not_registered")
}

@available(macOS 13.0, *)
func status() -> Outcome {
    let service = SMAppService.agent(plistName: agentPlist)
    var state = stateName(service.status)
    if service.status == .enabled, probeDaemonPort() == .prometheus { state = "running" }
    return Outcome(code: .ok, state: state)
}

// MARK: - The one-time pairing secret

/// The user's home directory, resolved the way the daemon resolves it (Python's Path.home() reads $HOME).
/// The launcher and the daemon must agree on this or they would look for the pairing secret and the logs in
/// different places; FileManager's own answer comes from the password database and ignores $HOME. For a
/// person's login the two are the same directory, so this changes nothing in production and lets a test run
/// the real launcher against an isolated home.
func homeDirectory() -> URL {
    if let home = ProcessInfo.processInfo.environment["HOME"], home.hasPrefix("/"), home.count > 1 {
        return URL(fileURLWithPath: home, isDirectory: true)
    }
    return FileManager.default.homeDirectoryForCurrentUser
}

/// ~/Library/Application Support/Prometheus/pairing/pair.secret: 32 random bytes, base64url, one line.
/// The directory is 0700 and the file 0600. Beacon reads it and posts it to the daemon's loopback
/// pairing route; the daemon compares against this file on every attempt and deletes it on first use.
func pairingDirectory() -> URL {
    homeDirectory().appendingPathComponent("Library/Application Support/Prometheus/pairing", isDirectory: true)
}

func randomSecret() -> String? {
    var bytes = [UInt8](repeating: 0, count: 32)
    guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else { return nil }
    return Data(bytes).base64EncodedString()
        .replacingOccurrences(of: "+", with: "-")
        .replacingOccurrences(of: "/", with: "_")
        .replacingOccurrences(of: "=", with: "")
}

/// Write a fresh secret, replacing any earlier one atomically so a reader never sees half a file.
func writePairSecret() -> Outcome {
    let fileManager = FileManager.default
    let directory = pairingDirectory()
    do {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    } catch {
        return Outcome(code: .failed, state: "pair_failed", detail: "could not prepare \(directory.path): \(error.localizedDescription)")
    }
    guard let secret = randomSecret() else {
        return Outcome(code: .failed, state: "pair_failed", detail: "no random bytes available")
    }
    let temp = directory.appendingPathComponent(".pair.secret.\(getpid())")
    let target = directory.appendingPathComponent("pair.secret")
    let descriptor = open(temp.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
    guard descriptor >= 0 else {
        return Outcome(code: .failed, state: "pair_failed", detail: "could not create \(temp.path) (errno \(errno))")
    }
    let line = Array((secret + "\n").utf8)
    let written = write(descriptor, line, line.count)
    close(descriptor)
    guard written == line.count else {
        unlink(temp.path)
        return Outcome(code: .failed, state: "pair_failed", detail: "short write to \(temp.path)")
    }
    guard rename(temp.path, target.path) == 0 else {
        unlink(temp.path)
        return Outcome(code: .failed, state: "pair_failed", detail: "could not move the secret into place (errno \(errno))")
    }
    return Outcome(code: .ok, state: "pair_secret_written", extra: ["path": target.path])
}

// MARK: - Uninstall

func removeIfPresent(_ url: URL, removed: inout [String], failed: inout [String]) {
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    do {
        try FileManager.default.removeItem(at: url)
        removed.append(url.path)
    } catch {
        failed.append("\(url.path): \(error.localizedDescription)")
    }
}

/// Unregister, remove the pairing secret, and move this app to the Trash. Data stays unless asked:
/// the default `~/.prometheus`, `~/.config/prometheus/env` and the logs are the person's conversations
/// and keys. A custom PROMETHEUS_CONFIG_DIR is never touched, because this process cannot know it.
@available(macOS 13.0, *)
func uninstall(purgeData: Bool) -> Outcome {
    let fileManager = FileManager.default
    let home = homeDirectory()
    var removed: [String] = []
    var failed: [String] = []
    var kept: [String] = []

    let stopped = unregister()
    if !stopped.ok { failed.append("unregister: \(stopped.detail)") }

    removeIfPresent(pairingDirectory(), removed: &removed, failed: &failed)
    // Leave no empty "Prometheus" folder behind. rmdir only removes an empty directory, so anything
    // else the person put there survives.
    rmdir(pairingDirectory().deletingLastPathComponent().path)

    let data = [
        home.appendingPathComponent(".prometheus"),
        home.appendingPathComponent(".config/prometheus"),
        home.appendingPathComponent("Library/Logs/Prometheus"),
    ]
    for url in data {
        if purgeData {
            removeIfPresent(url, removed: &removed, failed: &failed)
        } else if fileManager.fileExists(atPath: url.path) {
            kept.append(url.path)
        }
    }

    // Last: this process is running from the bundle it moves. macOS lets a running app be trashed.
    do {
        try fileManager.trashItem(at: Bundle.main.bundleURL, resultingItemURL: nil)
        removed.append(Bundle.main.bundleURL.path)
    } catch {
        failed.append("trash \(Bundle.main.bundleURL.path): \(error.localizedDescription)")
    }

    let extra: [String: Any] = ["removed": removed, "kept": kept, "failed": failed]
    if failed.isEmpty {
        return Outcome(code: .ok, state: "uninstalled", extra: extra)
    }
    return Outcome(code: .failed, state: "uninstall_incomplete", detail: failed.joined(separator: "; "), extra: extra)
}

// MARK: - --run: what launchd starts

/// Prepare the environment and replace this process with the bundled Python running the daemon.
/// The bundled plist is static, so per-user paths (logs, working directory) are set here.
func run() -> Never {
    let fileManager = FileManager.default
    let home = homeDirectory()
    let resources = Bundle.main.resourceURL ?? home
    let python = resources.appendingPathComponent("python/bin/python3.12").path

    guard fileManager.isExecutableFile(atPath: python) else {
        FileHandle.standardError.write(Data("Prometheus: bundled Python not found at \(python)\n".utf8))
        exit(Code.failed.rawValue)
    }

    // stderr to a log file (early-boot output and tracebacks); stdout is the daemon's own rotating log.
    let logDir = home.appendingPathComponent("Library/Logs/Prometheus")
    try? fileManager.createDirectory(at: logDir, withIntermediateDirectories: true)
    let errLog = logDir.appendingPathComponent("launchd.err.log").path
    // A crash loop (KeepAlive restarts every ten seconds) would grow this file without bound. Keep at most
    // two files of 5 MiB: when the live one is over the cap at start, it becomes launchd.err.log.1.
    if let size = (try? fileManager.attributesOfItem(atPath: errLog))?[.size] as? Int, size > 5 * 1024 * 1024 {
        let previous = errLog + ".1"
        try? fileManager.removeItem(atPath: previous)
        try? fileManager.moveItem(atPath: errLog, toPath: previous)
    }
    let errFd = open(errLog, O_WRONLY | O_CREAT | O_APPEND, 0o600)
    if errFd >= 0 { dup2(errFd, STDERR_FILENO) }
    let nullFd = open("/dev/null", O_WRONLY)
    if nullFd >= 0 { dup2(nullFd, STDOUT_FILENO) }

    fileManager.changeCurrentDirectoryPath(home.path)

    setenv("PROMETHEUS_INSTALL_KIND", "app", 1)
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1)   // the signed bundle is never written into
    setenv("PYTHONNOUSERSITE", "1", 1)          // nothing from ~/Library/Python leaks in
    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
    setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:\(home.path)/.local/bin:\(inheritedPath)", 1)

    let arguments = [python, "-m", "prometheus", "daemon", "--bind", "127.0.0.1"]
    let cArguments: [UnsafeMutablePointer<CChar>?] = arguments.map { strdup($0) } + [nil]
    execv(python, cArguments)
    // execv only returns on failure.
    FileHandle.standardError.write(Data("Prometheus: could not start the bundled Python (errno \(errno))\n".utf8))
    exit(Code.failed.rawValue)
}

// MARK: - The window (a person double-clicked the app)

/// Where Beacon is, if it is installed: asked of LaunchServices by bundle id, not by path.
func beaconApplication() -> URL? {
    NSWorkspace.shared.urlForApplication(withBundleIdentifier: beaconBundleID)
}

@available(macOS 13.0, *)
func showWindow() -> Never {
    let application = NSApplication.shared
    application.setActivationPolicy(.accessory)
    application.activate(ignoringOtherApps: true)

    func alert(_ title: String, _ text: String, buttons: [String], style: NSAlert.Style = .informational) -> Int {
        let box = NSAlert()
        box.messageText = title
        box.informativeText = text
        box.alertStyle = style
        for button in buttons { box.addButton(withTitle: button) }
        return box.runModal().rawValue - NSApplication.ModalResponse.alertFirstButtonReturn.rawValue
    }

    var message = ""
    while true {
        let current = status()
        let running = current.state == "running"
        let registered = running || current.state == "registered"
        let beacon = beaconApplication()

        if !registered {
            let text = current.state == "requires_approval"
                ? "Prometheus is switched off in Login Items. Turn it on to let it run in the background."
                : "Prometheus runs quietly in the background and starts when you log in. It listens on this Mac only."
            let choice = alert("Set up Prometheus", message.isEmpty ? text : message + "\n\n" + text,
                               buttons: ["Set up", "Quit"])
            if choice != 0 { exit(0) }
            let result = register()
            message = result.ok ? "" : (result.detail.isEmpty ? result.state : result.detail)
            continue
        }

        var buttons = [beacon == nil ? "Get Beacon" : "Open Beacon", "Pair Beacon again", "Uninstall…", "Quit"]
        let lead = running ? "Prometheus is running." : "Prometheus is set up and will start when you log in."
        let tail = beacon == nil
            ? "Prometheus needs Beacon to finish setup. Beacon is the app you talk to it with."
            : "Open Beacon to talk to it."
        let choice = alert(lead, (message.isEmpty ? "" : message + "\n\n") + tail, buttons: buttons)
        message = ""
        switch choice {
        case 0:
            if let beacon = beacon {
                NSWorkspace.shared.openApplication(at: beacon, configuration: NSWorkspace.OpenConfiguration())
            } else {
                NSWorkspace.shared.open(beaconURL)
            }
            exit(0)
        case 1:
            let result = writePairSecret()
            message = result.ok
                ? "Ready. Open Beacon and choose \"Connect to Prometheus on this Mac\"."
                : "Could not prepare pairing: \(result.detail)"
        case 2:
            // Cancel is added FIRST, so it is the default button: a stray Return (or anything that answers
            // the dialog for a person) cancels instead of uninstalling.
            let confirm = alert("Uninstall Prometheus?",
                                "This stops Prometheus and moves the app to the Trash. Your conversations, memory and keys in ~/.prometheus are kept.",
                                buttons: ["Cancel", "Uninstall"], style: .warning)
            if confirm == 1 {
                let result = uninstall(purgeData: false)
                _ = alert(result.ok ? "Prometheus is uninstalled." : "Uninstall did not finish.",
                          result.ok ? "Your data is still in ~/.prometheus." : result.detail, buttons: ["OK"])
                exit(result.code.rawValue)
            }
        default:
            exit(0)
        }
        buttons.removeAll()
    }
}

// MARK: - Entry

let arguments = Array(CommandLine.arguments.dropFirst())

if arguments.first == "--run" { run() }

@available(macOS 13.0, *)
func dispatch(_ arguments: [String]) -> Never {
    guard let mode = arguments.first else { showWindow() }
    switch mode {
    case "--register": finish(register())
    case "--unregister": finish(unregister())
    case "--status": finish(status())
    case "--pair": finish(writePairSecret())
    case "--uninstall":
        let flags = Set(arguments.dropFirst())
        guard flags.subtracting(["--purge-data"]).isEmpty else {
            finish(Outcome(code: .usage, state: "usage", detail: "Unknown option for --uninstall."))
        }
        finish(uninstall(purgeData: flags.contains("--purge-data")))
    default:
        finish(Outcome(code: .usage, state: "usage",
                       detail: "Unknown mode \(mode). Modes: --register --unregister --status --pair --uninstall --run"))
    }
}

if #available(macOS 13.0, *) {
    dispatch(arguments)
} else {
    // Built for macOS 11 so this branch can run and say so, instead of dyld refusing the launch.
    finish(Outcome(code: .unsupportedOS, state: "unsupported_os", detail: "Prometheus needs macOS 13 or later."))
}
