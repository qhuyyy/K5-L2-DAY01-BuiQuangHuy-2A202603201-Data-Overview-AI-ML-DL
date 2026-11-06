# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:06/11/2026**

**Runtime Colab:GPU T4** CPU/GPU

**Python / PyTorch / Ultralytics: Python 3**

**Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt,** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON: 0.35** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn: Không** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K".
2. Record này nói về **cả bức ảnh**. Bởi vì nó đang sử dụng model classification, dự đoán là cab với điểm cao nhất là 51,09%.
3. Danh sách tên lớp (class list) được xác nhận từ bộ nhãn (taxonomy) ImageNet-1K, không phải tự nhiên mà có.
4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name`, vì để tránh nhầm lẫn giữa các taxanomy khác nhau, đảm bảo tính nhất quán của dữ liệu, hỗ trợ mở rộng dataset.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ quy tắc chọn lớp đại diện, cụ thể như: quy tắc xác định đối tượng chính, quy tắc xử lý khi có nhiều đối tượng cùng/khác loại, quy tắc ưu tiên khi có nhiều chủ đề, quy tắc vi phạm nhãn
6. Bởi vì Score chỉ là kết quả dự đoán của model, không phải nhãn đúng. Ground truth phải được xác định độc lập với model prediction.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái toạ độ (385.33, 69.24), góc dưới-phải ở toạ độ (498.92, 348.92), rộng 113.58 pixel, cao 279.68 pixel
3. Chọn hai ngưỡng 0.20 và 0.60. Ngưỡng 0.20 có 17 vật thể, ngưỡng 0.60 có 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc như:

- Kiểm tra False Positive: Xác định những box không thực sự bao quanh đối tượng cần nhận diện.
- Kiểm tra Duplicate Boxes: Phát hiện nhiều box cùng bao quanh một đối tượng.
- Kiểm tra Class: Xác minh model có gán đúng nhãn cho đối tượng không.
- Kiểm tra Bounding Box: Xem box có bao trọn đối tượng hay bị lệch, thiếu hoặc thừa.
- Xử lý False Negative: Kiểm tra ảnh để tìm đối tượng thực tế mà model chưa phát hiện.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này): Vẽ box sát với phần nhìn thấy được của đối tượng, bao gồm tất cả phần đối tượng có thể quan sát, hạn chế đưa quá nhiều background vào box, không tự suy đoán phần bị che khuất hoàn toàn hoặc nằm ngoài khung hình.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Em sẽ tuỳ trường hợp để viết thành quy tắc trong guideline hay escalation. Nếu trường hợp quy tắc xử lý đã rõ và được áp dụng phổ biến, hay trường hợp đã được Lab Coach xác nhận, em sẽ viết thành quy tắc. Với những trường hợp phức tạp hơn, em sẽ trực tiếp hỏi thẳng Lab Coach. Tuy nhiên em sẽ luôn thiên về phương án escalation hơn bởi vì em là người mới, chưa có nhiều kinh nghiệm xử lý, nên để những người có kinh nghiệm xử lý và xác nhận trong những trường hợp đặc biệt.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-008" ; `class_name`="spoon" ; `score`=0.487579 ; số điểm polygon=`polygon_point_count`=46 ; 3–5 điểm đầu của `polygon_xy`=[
   544.0,
   67.0
   ],
   [
   544.0,
   80.0
   ],
   [
   545.0,
   81.0
   ],
   [
   545.0,
   91.0
   ],
   [
   544.0,
   92.0
   ],
2. So với box, polygon cho thêm thông tin về hình dạng thực tế của đối tượng, đường viền, diện tích và hình dáng vùng đối tượng, thông tin không gian chi tiết hơn bounding box
3. `instance_id` là mã định danh riêng cho từng đối tượng cụ thể trong ảnh. Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? Class ID dùng để xác đinh đối tượng, Tracking ID dùng để theo dõi cùng một đối tượng qua nhiều frame khác nhau. VD: 2 chiếc xe có thể có cùng class ID nhưng sẽ có instance ID khác nhau, instance ID dùng để phân biệt các đối tượng trong cùng một ảnh, nó không có nhiệm vụ nhận diện và duy trì danh tính một đối tượng xuyên suốt nhiều frame.
4. Viết **một quy tắc vẽ biên mask**: Vẽ theo đường viền bên ngoài của vật thể. Bao gồm toàn bộ phần nhìn thấy được. Không lấy pixel background vào mask. Không gộp hai đối tượng khác nhau thành một mask. Không tự vẽ thêm phần bị che khuất hoàn toàn.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Tuỳ trường hợp em sẽ dùng guideline hoặc escalation. Nhưng em sẽ thiên về escalation hơn vì muốn chắc chắn hơn về cách mình sẽ thực hiện, do bản thân chưa đủ kinh nghiệm.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| Phân loại ảnh | | JSON traffic: class_name = cab, score = 0.510915. Confidence chỉ khoảng 51%, cần kiểm tra ảnh có thực sự thuộc lớp này không. | Xác định chủ đề chính theo guideline và gán một class label. | --- | --- |
| Phân loại ảnh |  Một class label cho mỗi ảnh (single-label), ví dụ cab  | JSON traffic: class_name = cab, score = 0.510915. Confidence chỉ khoảng 51%, cần kiểm tra ảnh có thực sự thuộc lớp này không | Xác định chủ đề chính theo guideline và gán một class label | Kiểm tra class có đúng không, ảnh nhiều chủ thể có được xử lý nhất quán không |
| Phát hiện vật thể | Một bounding box (x1, y1, x2, y2) và class label cho mỗi đối tượng | JSON kitchen: box person, confidence 0.912625. Cần kiểm tra box có bao sát người không, có thiếu phần cơ thể hoặc chứa background quá nhiều không | Vẽ box cho từng đối tượng, gán class tương ứng | Kiểm tra class, vị trí box, kích thước, đối tượng bị bỏ sót và box trùng lặp |
| Instance segmentation | Một mask/polygon và class label cho mỗi instance | Khó khăn có thể nằm ở biên mờ, hai vật chạm nhau hoặc vùng bị che khuất. JSON prediction cần có dữ liệu polygon/mask để kiểm tra cụ thể | Vẽ mask riêng cho từng instance, bám theo phần nhìn thấy của đối tượng | Kiểm tra đường biên, pixel background lọt vào mask, mask bị thiếu và các instance bị gộp nhầm |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là chỉ sử dụng dữ liệu được cung cấp cho mục đích thực hiện bài lab, không tự ý chia sẻ, tải lên nền tảng bên ngoài hoặc tiết lộ thông tin nhạy cảm.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, em sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu, và báo cho Lab Coach hoặc Project Lead để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
