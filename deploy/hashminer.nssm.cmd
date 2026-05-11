@echo off
nssm install hashminer "C:\hashminer\hashminer.exe" --config "C:\hashminer\config.toml"
nssm set hashminer AppDirectory "C:\hashminer"
nssm set hashminer AppEnvironmentExtra "KEYSTORE_PASSWORD=__SET_ME__"
nssm set hashminer AppStdout "C:\hashminer\logs\stdout.log"
nssm set hashminer AppStderr "C:\hashminer\logs\stderr.log"
nssm set hashminer AppRotateFiles 1
