# Hướng dẫn kiểm tra nhãn khuôn mặt VTV (chuẩn CK+)

Thư mục này do `python -m src.data.vtv_faces` tạo ra. Mỗi ảnh PNG là **một khuôn mặt**, đã được xử lý giống ảnh CK+ 48×48:

1. **Phát hiện mặt**: YuNet (OpenCV) tìm khuôn mặt và 5 điểm mốc: 2 mắt, đầu mũi, 2 khóe miệng.
2. **Căn chỉnh theo CK+**: xoay, phóng và dịch ảnh sao cho 5 điểm mốc rơi đúng vị trí trung bình của chúng trong 927 ảnh CK+ 48×48. Vì vậy mắt luôn nằm ngang, ở cùng độ cao, và khung cắt giống hệt CK+.
3. **Trắng đen**, lưu ở 192×192 cho dễ nhìn. Bước dựng dataset sẽ thu nhỏ về **48×48** giống CK+.
4. **Chỉ giữ mặt rõ**. Các ngưỡng lấy theo chính biên độ đo được trên CK+:
   - Điểm tin cậy của YuNet ≥ 0.80.
   - Khoảng cách 2 mắt ≥ 24 px trong frame gốc. Mặt chỉ bị thu nhỏ khi đưa về 48×48, không bao giờ bị phóng to.
   - Nhìn gần thẳng: quay ngang ≤ ~15°, ngẩng/cúi trong khoảng của CK+, nghiêng ≤ 30° (độ nghiêng được xoay thẳng lại).
   - Không bị cắt ở mép khung hình.
   - Đủ sáng, đủ tương phản, đủ nét. Độ nét được đo ở 2 cỡ: 48×48 và 96×96 (chỉ số blur effect, không phụ thuộc độ sáng). Ảnh phim cũ nâng độ phân giải lên và ảnh bị lệch nét đều bị loại. Ảnh hơi mềm mà vẫn đọc được biểu cảm thì được giữ; nếu thấy nhòe thì bạn loại.
   - **Kiểm tra lại**: chạy YuNet lần 2 trên ảnh đã căn chỉnh. Phải tìm lại được mặt và 5 điểm mốc phải khớp khuôn CK+; bàn tay, đồ vật và mặt bị che nửa sẽ bị loại ở bước này.
5. **Bỏ ảnh trùng**: trong 60 giây của cùng một video, ảnh gần như y hệt ảnh đã giữ (NCC ≥ 0.85), hoặc khá giống (NCC ≥ 0.80) mà cùng nhãn gợi ý, thì bị bỏ. Biểu cảm đổi thì ảnh vẫn được giữ.
6. **Nhãn gợi ý** (`python -m src.data.vtv_relabel`) là trung bình của 3 model mã nguồn mở, **không phải model của nhóm**: FER+ (ONNX Model Zoo) và 2 model HSEmotion EfficientNet-B2 (train trên AffectNet). Mặt nào cả bộ tin ≥ 40% thì vào thư mục lớp, còn lại vào `_unsure/`. Độ chính xác đo trên ảnh có nhãn, cắt và căn y như ảnh ở đây:
   - **CK+**: 97% số mặt được xếp vào lớp, trong đó đúng **90%** (FER+ một mình chỉ đúng 66%, angry 13%, fear 4%).
   - **RAF-DB** (ảnh ngoài đời, giống phim nhất): 84% được xếp, trong đó đúng **74%**.
   - Lớp còn yếu nhất là **fear** (hay lẫn với surprise) và **neutral** (mặt biểu cảm nhẹ hay bị cho là neutral). Hai thư mục này cần xem kỹ nhất.
   - Vẫn chỉ là gợi ý: **mọi ảnh đều cần người duyệt**.

## Cách làm

**Cách nhanh (nên dùng): trang duyệt.** Bấm đúp `duyet_nhan_vtv.bat` ở thư mục project, hoặc chạy `python -m src.data.vtv_review_app`. Trình duyệt mở `http://127.0.0.1:8765` (chỉ máy này truy cập được).

- Chọn thư mục ở hàng trên (nên bắt đầu với disgust, fear, angry, surprise). Ảnh model ít chắc nhất hiện trước.
- Click hoặc dùng phím mũi tên chọn ảnh **sai** → bấm phím lớp đúng: `1` angry, `2` disgust, `3` fear, `4` happy, `5` neutral, `6` sad, `7` surprise, `U` chưa chắc, `X` loại. Ctrl/Shift + click để chọn nhiều ảnh rồi bấm một phím.
- **Enter** lưu cả trang: ảnh đổi lớp được chuyển thư mục; ảnh không đổi được ghi là **đã xác nhận đúng**. Vì vậy hãy nhìn hết trang trước khi bấm Enter.
- Ô viền nét đứt **"Claude → …"** là đề xuất của Claude (đã xem trước disgust, fear, surprise, angry). Đồng ý thì để nguyên; không đồng ý thì chọn ô và bấm phím của thư mục hiện tại để giữ, hoặc phím lớp đúng. Claude chấm mờ khá chặt và nhìn kém hơn model, nên ảnh hơi mềm mà vẫn đọc được biểu cảm thì cứ giữ. Phím **C** bật/tắt đề xuất.
- **Ctrl+Z** hoàn tác trang vừa lưu, **Space** phóng to ảnh (kèm ảnh 48×48 mà model học), **?** xem bảng tiêu chí.
- Mọi quyết định được ghi vào `data/vtv/human_review.csv`. Bước dựng dataset dựa vào file này để biết nhãn nào đã được người duyệt, nên đừng xóa nó.
- Tắt trang (đóng cửa sổ đen) trước khi build lại dataset: pipeline không chạy khi trang duyệt đang mở.

**Cách cũ: File Explorer.** Vẫn dùng được, nhưng chỉ ảnh bị chuyển thư mục mới được tính là đã duyệt.

- Mở từng thư mục trong File Explorer, chọn View → Large icons (hoặc Extra large icons).
- Ảnh nào sai lớp thì **cắt (Ctrl+X) và dán sang đúng thư mục lớp**.
- **Không đổi tên file** (tên là mã nối với `faces.csv`) và **không sửa ảnh**.
- `_unsure/`: chưa chắc, cần người thứ hai xem. Ảnh còn lại ở đây sẽ **không** vào dataset.
- `_reject/`: ảnh hỏng, xem mục "Loại bỏ" bên dưới. Ảnh ở đây sẽ **không** vào dataset.
- Làm xong, chạy lại `python -m src.data.vtv_build` (hoặc báo Claude). Lệnh này dựng lại `data/vtv/dataset/`: thu về 48×48 và chia train/val/test **theo người** (nhận dạng mặt bằng SFace), giống cách CK+ chia theo người. Một diễn viên chỉ nằm trong một tập, nên điểm test không bị ảo. Lệnh cũng kiểm tra cân bằng lớp. Dataset dựng trước khi duyệt chỉ chứa nhãn gợi ý; `report.md` sẽ ghi rõ điều đó.

## Tiêu chí 7 lớp (theo CK+, dựa trên FACS Action Units)

Chỉ nhìn **khuôn mặt**, không đoán theo bối cảnh phim hay phụ đề. CK+ gán nhãn theo các cơ mặt (AU) bắt buộc phải có:

| Thư mục | Bắt buộc phải thấy (CK+) | Dấu hiệu dễ nhìn |
|---|---|---|
| `angry` | AU23/AU24: môi mím chặt hoặc ép vào nhau, thường kèm AU4 (chau mày) | Lông mày kéo xuống và chụm lại, mắt trừng hoặc nheo, môi mím hoặc nghiến răng |
| `disgust` | AU9 (nhăn mũi) hoặc AU10 (nâng môi trên) | Sống mũi nhăn, môi trên nhếch lên, má nâng |
| `fear` | AU1+2+4: lông mày vừa nhướn lên vừa chụm lại; mí trên mở to (AU5) | Mắt mở to, lông mày nhướn và chau, miệng kéo ngang sang hai bên (AU20), có thể hé mở |
| `happy` | AU12: khóe môi kéo lên | Cười; má nâng, đuôi mắt nhăn (AU6) |
| `neutral` | Không có cơ mặt nào hoạt động rõ | Mặt thả lỏng: không cười, không chau mày, miệng khép tự nhiên |
| `sad` | AU1+4+15 (đầu trong lông mày nhướn, chau, khóe môi trễ xuống) hoặc AU11 | Đầu trong lông mày (gần sống mũi) nhướn lên thành hình mái nhà /\, khóe miệng trễ xuống |
| `surprise` | AU1+2 (nhướn cả lông mày) hoặc AU5 (mở to mắt), thường kèm há miệng (AU26/27) | Lông mày nhướn cao, mắt mở to, cằm hạ, miệng há tròn |

Lưu ý khi phân biệt:

- **Đang nói ≠ ngạc nhiên.** Miệng há vì đang nói, còn lông mày và mắt bình thường → `neutral`. Nếu không chắc → `_unsure`.
- **Fear và surprise** đều nhướn mày, mở to mắt. Fear có thêm **chau mày** (AU4) và môi **kéo ngang**; surprise thì mày cong cao, không chau, miệng há tròn.
- **Angry và disgust**: angry là chau mày và mím môi; disgust là **nhăn mũi, nhếch môi trên**.
- **Sad và neutral**: sad cần khóe môi trễ hoặc đầu trong lông mày nhướn lên; chỉ cúi mặt hay nhìn xuống thì vẫn là `neutral`.
- Khinh bỉ (contempt, nhếch một bên mép, AU14): CK+ có lớp này nhưng project không dùng → `_reject`.
- Chọn ảnh có biểu cảm **rõ**, giống ảnh đỉnh điểm trong CK+. Biểu cảm quá nhẹ, không chắc → `_unsure`.

## Loại bỏ (`_reject`)

- Không phải mặt người: tranh, poster, hình in, búp bê, tượng.
- Bị che: tay, micro, khẩu trang, tóc hoặc kính đen che mắt hay miệng; chữ hoặc logo đè lên mặt.
- Quá mờ, nhòe chuyển động, hoặc quá tối không nhìn rõ mắt và miệng.
- Căn chỉnh hỏng: mắt không nằm ngang ở khoảng 40% chiều cao ảnh, mặt lệch hẳn sang một bên, hoặc khung cắt chứa hai mặt.
- Đang nhắm mắt vì chớp mắt (không phải biểu cảm).
- Mặt quay nghiêng rõ (thấy một bên má nhiều hơn hẳn bên kia).
