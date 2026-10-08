#!/bin/bash
# set_release_secrets.sh: set the five repository secrets .github/workflows/release-macos.yml needs, in one go.
#
#   packaging/macos/set_release_secrets.sh [--repo OAraLabs/Prometheus] [--dry-run]
#
# It asks for the values and never prints them. They go straight to `gh secret set` on stdin, so they are not
# in argv, the shell history, or a file.
#
# Why per repository: the owner (OAraLabs) is a user account, not an organization, so there are no
# organization secrets to share, and GitHub never lets a secret be read back. beacon-desktop's copies cannot
# be copied from there; the SAME VALUES (same Developer ID Application certificate, same Apple ID, same
# team) are typed in again for this repo.
#
#   CSC_LINK                     the Developer ID Application certificate and its private key, as a .p12,
#                                base64 encoded. The file must hold exactly ONE Developer ID Application
#                                identity of team 53JM8W47RL (the workflow stops if it finds two; a keychain
#                                export of both same-named certificates would).
#   CSC_KEY_PASSWORD             that .p12's password
#   APPLE_ID                     the Apple ID's email
#   APPLE_APP_SPECIFIC_PASSWORD  an app-specific password from appleid.apple.com (xxxx-xxxx-xxxx-xxxx).
#                                A NEW one named for this repo is better than reusing Beacon's: revoking one
#                                then does not break the other.
#   APPLE_TEAM_ID                53JM8W47RL
#
# Not needed here (beacon-desktop has it, this repo does not): BEACON_RELEASES_TOKEN. Beacon publishes to a
# different repository; this workflow attaches to a draft release in its own repo with the job's token.

REPO="OAraLabs/Prometheus"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "usage: $0 [--repo OWNER/NAME] [--dry-run]"; exit 2 ;;
  esac
done

set_secret() {  # name, value (via stdin)
  if [ "$DRY" -eq 1 ]; then
    n=$(wc -c | tr -d ' '); echo "  [dry run] would set $1 on $REPO ($n bytes)"
  else
    gh secret set "$1" --repo "$REPO" >/dev/null && echo "  set $1"
  fi
}

[ "$DRY" -eq 1 ] || { gh auth status >/dev/null 2>&1 || { echo "gh is not signed in: run 'gh auth login' first."; exit 1; }; }
echo "Setting release secrets on $REPO. Values are not shown."

read -r -p "Path to the Developer ID Application .p12: " P12
P12="${P12/#\~/$HOME}"
[ -s "$P12" ] || { echo "no such file: $P12"; exit 1; }
read -r -s -p ".p12 password: " P12PASS; echo

# Count the identities in the file (client certificates with a key), by name, without putting the password in argv.
ids="$(printf '%s' "$P12PASS" | openssl pkcs12 -in "$P12" -passin stdin -clcerts -nokeys 2>/dev/null \
      | openssl x509 -noout -subject 2>/dev/null | grep -c 'Developer ID Application')"
total="$(printf '%s' "$P12PASS" | openssl pkcs12 -in "$P12" -passin stdin -clcerts -nokeys 2>/dev/null | grep -c 'BEGIN CERTIFICATE')"
if [ "$total" -eq 0 ]; then echo "could not open the .p12: wrong password, or not a .p12."; exit 1; fi
echo "  the file holds $total certificate(s)."
[ "$total" -eq 1 ] || { echo "It must hold exactly one identity (it holds $total). Export only one, by its fingerprint."; exit 1; }
printf '%s' "$P12PASS" | openssl pkcs12 -in "$P12" -passin stdin -clcerts -nokeys 2>/dev/null | openssl x509 -noout -subject -enddate | sed 's/^/  /'
case "$(printf '%s' "$P12PASS" | openssl pkcs12 -in "$P12" -passin stdin -clcerts -nokeys 2>/dev/null | openssl x509 -noout -subject 2>/dev/null)" in
  *"Developer ID Application"*53JM8W47RL*) ;;
  *) echo "that certificate is not a Developer ID Application of team 53JM8W47RL."; exit 1 ;;
esac

read -r -p "Apple ID (email): " APPLE_ID
read -r -s -p "App-specific password (xxxx-xxxx-xxxx-xxxx): " APPLE_PW; echo
read -r -p "Apple team ID [53JM8W47RL]: " TEAM; TEAM="${TEAM:-53JM8W47RL}"
case "$APPLE_PW" in [a-z][a-z][a-z][a-z]-[a-z][a-z][a-z][a-z]-[a-z][a-z][a-z][a-z]-[a-z][a-z][a-z][a-z]) ;; *) echo "  note: that is not the xxxx-xxxx-xxxx-xxxx shape of an app-specific password." ;; esac

base64 < "$P12" | tr -d '\n' | set_secret CSC_LINK
printf '%s' "$P12PASS" | set_secret CSC_KEY_PASSWORD
printf '%s' "$APPLE_ID" | set_secret APPLE_ID
printf '%s' "$APPLE_PW" | set_secret APPLE_APP_SPECIFIC_PASSWORD
printf '%s' "$TEAM" | set_secret APPLE_TEAM_ID
echo "Done. Check names (never values) with: gh secret list --repo $REPO"
