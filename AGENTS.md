# Repository Guidelines

## Project Structure & Module Organization

This repository builds Howdy, a Linux face-authentication stack. The root `meson.build` wires together:

- `howdy/src/`: core CLI, face comparison, camera recorders, PAM scripts, config template, and assets.
- `howdy/src/pam/`: C++ PAM module and translation metadata.
- `howdy/src/cli/`: individual CLI commands such as `add`, `test`, `set`, and `remove`.
- `howdy-gtk/src/`: GTK UI code, Glade files, icons, and polkit policy template.
- `howdy/debian/`, `howdy-gtk/debian/`, and `howdy/archlinux/`: distro packaging files.

There is no dedicated `tests/` directory; verification is mostly build checks plus targeted manual CLI/PAM testing.

## Build, Test, and Development Commands

- `scripts/install-local.sh`: create/update a project-local `.venv` with UV, install backend dependencies for the current branch, download required backend model files, build with Meson, install to `/usr/local`, install the PAM module to the path referenced by `/etc/pam.d`, and verify the installed backend.
- `meson setup build -Dpython_path=/path/to/python`: configure a local build. Use the interpreter with the required backend stack.
- `meson compile -C build`: compile the C++ PAM module and generate configured files.
- `meson install -C build`: install to the configured prefix. This can affect authentication paths, so review options first.
- `meson configure build`: inspect active build options such as `config_dir`, `user_models_dir`, and `log_path`.
- `meson setup --wipe build ...`: recreate an existing build directory after changing options.

For PAM-only work, `howdy/src/pam/README.md` documents the smaller `meson setup build && meson compile -C build` flow.

## Coding Style & Naming Conventions

Python files use snake_case modules and functions, four-space indentation for new code, and module-level command dispatch in `howdy/src/cli.py`. Wrap user-facing strings with `_()` where surrounding code is localized. C++ in `howdy/src/pam/` uses two-space indentation, `auto ... -> type` signatures, standard-library types, and syslog/PAM return conventions. Match nearby code before adding abstractions.

## Testing Guidelines

No pytest, unittest, or coverage configuration is committed. Before submitting changes, run `meson compile -C build`. For behavior changes, manually exercise the relevant command, for example `sudo howdy test`, `sudo howdy add`, or `sudo howdy list`, and check `/var/log/auth.log` when PAM behavior is involved. Avoid authentication changes without testing failure paths.

## Maintenance Notes

When project behavior, dependencies, install paths, or backend requirements change, update `AGENTS.md` in the same change. If the change affects setup, Python dependencies, model files, Meson options, install locations, or backend detection, update `scripts/install-local.sh` as well.

## Commit & Pull Request Guidelines

Recent commits use short, informal imperative summaries such as `remove dlib support` and `threaded deepface import`. Keep subjects concise and focused on one change. Pull requests should describe user-visible behavior, list build/manual tests, mention affected platforms or packaging files, and include screenshots for GTK UI changes. Link related issues when applicable.

## Security & Configuration Tips

Changes can affect login, sudo, and PAM authentication. Treat defaults in `howdy/src/config.ini` and path options in `meson.options` as security-sensitive. Do not commit face models, logs, virtualenv paths, or machine-specific camera settings.
