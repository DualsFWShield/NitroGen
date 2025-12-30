@echo off
title NitroGen Launcher
color 0A

:: --- CONFIGURATION ---
set PROJECT_DIR=Your path to NitroGen
set MODEL_FILE=ng.pt
set TIMESTEPS=12
:: Vous pouvez changer les parametres ici
set ARGS=--compile --ctx 1 --cfg 1.0 --no-cache
:: ---------------------

:: 1. Aller dans le dossier du projet
if exist "%PROJECT_DIR%" (
    cd /d "%PROJECT_DIR%"
) else (
    echo ERREUR: Le dossier %PROJECT_DIR% est introuvable.
    pause
    exit
)

echo ========================================================
echo                 NITROGEN AUTO-LAUNCHER
echo ========================================================
echo.

:: 2. Lancer le serveur d'inference
echo [1/3] Lancement du serveur (Timesteps: %TIMESTEPS%)...
:: On utilise start pour lancer une nouvelle fenetre
start "NitroGen Server" cmd /k "python scripts\serve.py %MODEL_FILE% --timesteps %TIMESTEPS% %ARGS%"

:: 3. Attente intelligente
echo [2/3] Attente du chargement du modele (5 secondes)...
timeout /t 5 >nul

echo.
echo ========================================================
echo                 CONFIGURATION DU JEU
echo ========================================================

:: --- LOGIQUE DE MEMORISATION DU JEU ---
set "PREV_GAME="
if exist "last_game.txt" (
    set /p PREV_GAME=<last_game.txt
)

if defined PREV_GAME (
    echo Dernier jeu utilise : [%PREV_GAME%]
    echo Appuyez sur ENTREE pour reutiliser ce jeu, ou tapez un nouveau nom.
) else (
    echo Entrez le nom exact de l'executable (ex: valorant.exe)
)

set /p GAME_INPUT="Nom de l'executable : "

:: Si l'utilisateur appuie juste sur Entrée et qu'il y a un jeu precedent, on l'utilise
if "%GAME_INPUT%"=="" set GAME_INPUT=%PREV_GAME%

:: Retirer les guillemets eventuels
set GAME_EXE=%GAME_INPUT:"=%

:: Sauvegarder pour la prochaine fois
echo %GAME_EXE% > last_game.txt

:: 4. Lancer l'agent
echo.
echo [3/3] Lancement de l'agent pour : %GAME_EXE% ...
echo.
echo ========================================================
echo    L'AGENT EST EN COURS D'EXECUTION
echo    Appuyez sur Ctrl+C dans CETTE fenetre pour arreter.
echo ========================================================
echo.

python scripts\play.py --process "%GAME_EXE%"

:: 5. Section de fermeture propre (Kill Switch manuel)
echo.
echo ========================================================
echo L'agent est arrete. Voulez-vous fermer le serveur aussi ?
echo (Cela tuera tous les processus python lies a NitroGen)
echo ========================================================
pause
taskkill /FI "WINDOWTITLE eq NitroGen Server*" /T /F
echo Serveur ferme. A bientot !
timeout /t 2 >nul