@echo off
chcp 65001 >nul
echo ========================================
echo   FrFNO GitHub Upload Script
echo ========================================
echo.

cd /d "D:\科研\FrFNO_release"

:: Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Please install Git first:
    echo         https://git-scm.com/download/win
    pause
    exit /b 1
)

:: Init repo
if not exist .git (
    echo [1/5] Initializing git repository...
    git init
    git branch -M main
) else (
    echo [1/5] Git repository already exists.
)

:: Configure user (if not set)
git config user.name >nul 2>&1
if errorlevel 1 (
    set /p GIT_NAME=Enter your GitHub username: 
    set /p GIT_EMAIL=Enter your GitHub email: 
    git config user.name "%GIT_NAME%"
    git config user.email "%GIT_EMAIL%"
)

:: Add files
echo [2/5] Adding files...
git add .
git commit -m "Initial release: FrFNO reproduction code" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Nothing to commit (already up to date).
)

:: Check remote
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo.
    echo [3/5] No remote repository configured.
    echo.
    echo Please create a new repository on GitHub:
    echo   1. Go to https://github.com/new
    echo   2. Repository name: FrFNO
    echo   3. Do NOT check "Add a README file"
    echo   4. Click "Create repository"
    echo.
    set /p REPO_URL=Paste your repository URL (e.g. https://github.com/username/FrFNO.git): 
    git remote add origin "%REPO_URL%"
) else (
    echo [3/5] Remote already configured.
)

:: Push
echo [4/5] Pushing to GitHub...
git push -u origin main
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Common causes:
    echo   - Authentication failed (use GitHub Personal Access Token as password)
    echo   - Repository not created on GitHub
    echo   - URL incorrect
    echo.
    pause
    exit /b 1
)

echo [5/5] Done! Your code is now on GitHub.
echo.
pause
