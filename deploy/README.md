# hashminer deployment

## Linux (systemd)

1. Copy `hashminer` binary to `/opt/hashminer/hashminer`
2. Copy `config.example.toml` to `/etc/hashminer/config.toml` and edit (RPC URL, keystore path)
3. Create `/etc/hashminer/env` with `KEYSTORE_PASSWORD=...`
4. Create unprivileged `miner` user that owns `/opt/hashminer`
5. `sudo cp hashminer.service /etc/systemd/system/ && sudo systemctl enable --now hashminer`
6. `journalctl -u hashminer -f` for live logs

## Windows (NSSM)

1. Install [NSSM](https://nssm.cc/), place `hashminer.exe` in `C:\hashminer\`
2. Edit `hashminer.nssm.cmd` to set the real `KEYSTORE_PASSWORD`
3. Run `hashminer.nssm.cmd` as administrator
4. `nssm start hashminer`
