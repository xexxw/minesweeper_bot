@echo off
echo Building minesweeper_bot.exe ...
pyinstaller --onefile --noconsole minesweeper_bot.py
echo Done! Check dist\minesweeper_bot.exe
pause
