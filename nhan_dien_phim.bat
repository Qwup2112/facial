@echo off
rem Nhan dien cam xuc tren phim (src.demo.webcam): chon file video, cua so hien khung mat + cam xuc.
rem Phim tat: Space dung/chay, mui ten trai/phai lui/toi 10 giay, S chup man hinh (reports\demo), Q hoac Esc thoat.
rem Khung xam = mat nghieng/mo/nho, model chua hoc kieu mat nay nen nhan chi la doan.
rem Webcam: python -m src.demo.webcam  (khong co --pick)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Python314\python.exe" -m src.demo.webcam --pick %*
pause
