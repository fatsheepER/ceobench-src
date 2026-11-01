# Registered script legacy client fixtures

Frozen public artifacts from commit `23a90ff`, before workspace-relative script registration and automatic observation-only permissions. They contain no credentials or world data. `test_registered_script_legacy.py` installs them before creating a checkpoint, then drives the actual restored public client.

| Fixture | Original path | SHA-256 |
| --- | --- | --- |
| `registered-script-client-23a90ff.pyz` | `public/novamind-client` | `03539edaeb86cd558363925c5ab1f539125cd89bf77c747d5c9257f099fb1a24` |
| `registered-script-cli-23a90ff.md` | `public/docs/cli.md` | `ce35e2b9c2bb601d224925df121333f75db84980119db62c08900bdc3bc45d5c` |

Regenerate each fixture with `git show 23a90ff:ORIGINAL_PATH > FIXTURE_PATH`. Preserve bytes; do not rebuild the legacy fixture with current source.
