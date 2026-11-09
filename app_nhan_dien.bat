@echo off
rem App nhan dien cam xuc (giao dien moi): mo video / anh tuy y, tua, nhap moc thoi gian, danh dau moc, quet ca phim.
rem Mo cua so Chrome rieng; dong cua so la may chu tu tat.   Tham so them, vi du:  app_nhan_dien.bat --run runs\transfer_mix\<run>
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Python314\python.exe" -m src.demo.app %*
if errorlevel 1 pause
