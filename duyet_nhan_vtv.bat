@echo off
rem Trang duyet nhan VTV (src.data.vtv_review_app): mo http://127.0.0.1:8765 trong trinh duyet.
rem Dong cua so nay (hoac Ctrl+C) khi duyet xong, truoc khi build lai dataset.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Python314\python.exe" -m src.data.vtv_review_app %*
pause
