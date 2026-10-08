# Clean-Mac test: Prometheus for Mac

About 20 minutes, on a Mac that has never had Prometheus. **Don't install anything first** (no Homebrew, no
Python, no Xcode tools): the point is that none of it is needed. Needs an Apple-silicon Mac (M1 or later) on
macOS 13 or newer: Apple menu ▸ About This Mac shows both.

You need two files from Will: `Prometheus-<version>-arm64.zip` and `clean_mac_check.sh`. Send them by
AirDrop (that also puts the normal "downloaded file" safety check on them, which is the realistic case).

If anything looks wrong: **stop, screenshot it, write down which step**, and carry on only if you can. Don't
fix anything.

1. **Start a stopwatch.**
2. Double-click the zip. A `Prometheus` app appears next to it.
3. In Finder choose Go ▸ Go to Folder… and type `~/Applications`. (If it says the folder isn't found, make it:
   File ▸ New Folder, name it `Applications`, inside your home folder.) Drag **Prometheus** into it.
4. Open **Terminal** (Spotlight: type "Terminal"). Run these two lines (adjust the path if the script
   went somewhere other than Downloads):
   ```
   chmod +x ~/Downloads/clean_mac_check.sh
   ~/Downloads/clean_mac_check.sh before
   ```
   Every line should say PASS (lines starting `info` are just facts). Apple's check must say
   "Notarized Developer ID". If macOS pops up a window offering to install developer tools, click
   **Not Now** and tell Will.
5. **Double-click Prometheus** in `~/Applications`. (Don't right-click ▸ Open: that skips the very check we
   are testing.) The most you should see is the ordinary "Prometheus is an app downloaded from the
   Internet. Are you sure?" box: click **Open**. You should NOT see "unidentified developer" or "can't be
   opened". Screenshot whatever appears.
6. A window **Set up Prometheus** appears. Click **Set up**. macOS should show a "Background Items Added"
   notice for Prometheus. If it instead says it needs approval, go to System Settings ▸ General ▸
   Login Items & Extensions, turn **Prometheus** on, and open the app again. **Stop the stopwatch when the
   window says "Prometheus is running."** Write the time down.
7. In Terminal: `~/Downloads/clean_mac_check.sh running`. Then open System Settings ▸ General ▸
   Login Items & Extensions and **screenshot the Prometheus entry** (its name and its icon).
8. `~/Downloads/clean_mac_check.sh killtest` (stops it on purpose; it should come back by itself in about
   10 seconds).
9. **Log out and back in** (Apple menu ▸ Log Out). Wait 30 seconds after logging in, then
   `~/Downloads/clean_mac_check.sh running` again. It should be running without you opening anything.
10. From **another device on the same Wi-Fi** (your phone's browser is fine), open
    `http://<this Mac's address>:8005`. The address is the one the `running` check printed. It must fail
    to connect.
11. `~/Downloads/clean_mac_check.sh uninstall`, then look at Login Items & Extensions again: the
    Prometheus entry should be gone, and the app should be in the Trash.
12. Send Will: the file **`~/Desktop/prometheus-test-report.txt`**, every screenshot, and your stopwatch time.

What this checks: the app opens on a normal Mac with no warning, it runs only on this Mac, it shows up as
"Prometheus" in Login Items, it comes back after a crash and after logging in, and it removes itself cleanly.
It does not check Beacon's part (the "Set up Prometheus on this Mac" button), which is not built yet.
