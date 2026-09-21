# OpenCode V1 → V2 Final (Windows rescue guide)

Use this if the TUI session dies. Run in a **fresh PowerShell outside OpenCode**.
V2 final is `2.0.6`, package `@opencode/cli`, binary `opencode` (same name as V1).
No npm. pnpm only. Use `pnpm.cmd` to bypass the `ExecutionPolicy` block on `.ps1`
without changing system policy.

## 0. Close OpenCode first

```powershell
# Inside OpenCode TUI:
/exit
```

Then open a new Windows PowerShell. Confirm no opencode process holds the winget exe:

```powershell
Get-Process -Name "opencode" -ErrorAction SilentlyContinue
winget list --name "opencode"
opencode --version
```

Expected before migration: `SST.opencode` `1.18.30`, `opencode --version` → `1.18.x`.

## 1. Fix pnpm PATH for this shell

`pnpm setup` already set persistent `PNPM_HOME`, but you need a new terminal
or this one-liner per shell:

```powershell
$env:PNPM_HOME = "C:\Users\lucas\AppData\Local\pnpm"
$env:PATH = "C:\Users\lucas\AppData\Local\pnpm\bin;" + $env:PATH
& "C:\Users\lucas\AppData\Local\pnpm\bin\pnpm.cmd" --version
```

Expected: `12.4.1` or newer.

## 2. Remove V1 winget (required, same binary name)

Do this **after quitting OpenCode**, otherwise you get:
`remove: Acceso denegado.: "...\opencode.exe"`.

```powershell
winget uninstall --id SST.opencode --silent
```

If uninstall asks for confirmation, re-run without `--silent`:

```powershell
winget uninstall --id SST.opencode
```

Verify removal:

```powershell
winget list --name "opencode"
Get-Command opencode -ErrorAction SilentlyContinue
```

## 3. Clean aborted V2 beta (safe to ignore errors)

The earlier `@opencode-ai/cli@next` (`opencode2`, `0.0.0-beta-17823`) download
was aborted mid-download:

```powershell
& "C:\Users\lucas\AppData\Local\pnpm\bin\pnpm.cmd" remove -g @opencode-ai/cli
```

## 4. Install V2 final via pnpm

```powershell
& "C:\Users\lucas\AppData\Local\pnpm\bin\pnpm.cmd" add -g --allow-build=@opencode/cli @opencode/cli
```

This is the current documented final install (`https://opencode.ai/v2/docs`).

## 5. Verify V2

Close and reopen PowerShell once so `PNPM_HOME\bin` loads persistently, then:

```powershell
opencode --version
where.exe opencode
opencode debug paths
```

Expected: `2.0.6` (or newer 2.x), path under `C:\Users\lucas\AppData\Local\pnpm\bin`.

## 6. Config already migrated

`opencode.json` in this project was already migrated V1 → V2 native:
`mcp.google-classroom` → `mcp.servers.google-classroom`.
`type`, `command`, `environment` unchanged (V2 still uses `environment`).
`$schema` preserved. No `.opencode/` agents/commands/skills existed to migrate.
Global `C:\Users\lucas\.config\opencode\opencode.jsonc` is just `{"$schema": ...}` — nothing to do.
`tui.json` left alone (V2 auto-migrates to `cli.json` on first run).

Verify after install:

```powershell
opencode debug config
opencode mcp list
```

Then start normally:

```powershell
opencode
```

## If pnpm still says `ERR_PNPM_GLOBAL_BIN_DIR_NOT_IN_PATH`

Run once, then open a **new** terminal:

```powershell
& "C:\Users\lucas\AppData\Local\pnpm\bin\pnpm.cmd" setup
```

## Do NOT run

```powershell
# Don't use — old beta, installs opencode2:
pnpm add -g --allow-build=@opencode-ai/cli @opencode-ai/cli@next

# Don't use — insecure per your policy, and conflicts with pnpm global:
npm install -g @opencode/cli
```
