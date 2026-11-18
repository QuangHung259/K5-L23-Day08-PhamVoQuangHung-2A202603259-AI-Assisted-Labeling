# Báo cáo Day 08 — 2A202603259 - Phạm Võ Quang Hưng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person` | YOLO26m (`yolo26m.pt`) | `0.50` | Ở `r01` (`conf=0.25`) có quá nhiều box `person` nhỏ chồng chéo ở hậu cảnh (`img_002`, `img_032`); quét ngưỡng trên 4 ảnh gold (`submission/qc/threshold_sweep.md`) cho thấy tại `0.50` recall vẫn đạt `13/14 = 92.9%` ($\ge 90\%$) và precision tăng lên `59.1%` (so với `35.1%` ở ngưỡng `0.10`). |
| `bicycle` / `bus` / `truck` | YOLO26m (`yolo26m.pt`) | `0.30` (mặc định) | Trong 4 ảnh gold chỉ có `3` `bicycle`, `0` `bus`, `1` `truck` (đều `< 5` vật tối thiểu của `threshold_sweep.py`). Mẫu quá ít để chọn ngưỡng riêng nên giữ ngưỡng mặc định `--conf 0.30`. |
| `motorbike` / `car` | YOLO26m (`yolo26m.pt`) | `0.30` (mặc định) | Trên 4 ảnh gold (`11` `motorbike`, `8` `car`), không có ngưỡng nào $\le 0.50$ đạt recall mục tiêu $90\%$ (tối đa đạt `9/11 = 81.8%` với `motorbike` và `7/8 = 87.5%` với `car`, xem `threshold_sweep.md`). Do đó giữ ngưỡng mặc định `0.30` thay vì nâng lên `0.50` làm sót xe như ở `r02`. |
| `three_wheeler` | YOLOE-26s-seg (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.15` | YOLO26 (`r01`, `r02`) hoàn toàn không có lớp `three_wheeler` trong 80 lớp COCO. YOLOE (`r03`, `r07`) phát hiện được cả tuk-tuk, xích lô và ba gác; theo `threshold_sweep.md`, ngưỡng `0.15` đạt recall `7/7 = 100%` và precision `7/9 = 77.8%` trên 7 đối tượng gold. |

- **Lệnh `merge_prelabels.py` đã dùng (dán nguyên):**
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
  *(Đặt `yoloe_all` trước `y26_all` để các box `three_wheeler` có ưu tiên cao hơn và loại bỏ các box `car`/`truck` của YOLO26 chồng lên cùng chiếc xe ba bánh khi $\text{IoU} \ge 0.6$; kết quả gộp giữ lại `510` box và loại được `11` box `car`, `6` box `truck`, `2` box `bus`, `1` box `bicycle` bị đè).*
- **Phiên bản SAM bạn chọn (SAM 1 / 2.1 / 3) và lý do:**
  - Chọn **SAM 2.1 base (`sam2.1_b.pt`, 154.4 MB)** làm model chính cho Part 5, và chạy đối chiếu thêm với **SAM 1 MobileSAM (`mobile_sam.pt`, 38.8 MB)** (`r04` vs `r05` trong `model_log.csv`).
  - **Lý do:** Dù `mobile_sam.pt` chạy nhanh hơn trên CPU (`25.50s/ảnh` ở `r05` so với `45.21s/ảnh` của `sam2.1_b.pt` ở `r04`), khi kiểm tra trên `img_013.jpg` (`tuktuk_img013_sam21.jpg` vs `tuktuk_img013_sam1.jpg`), `sam2.1_b.pt` tạo ra **1 polygon duy nhất liền mạch** (`317,255 px`, chiếm `45.4%` ảnh) bám sát mui xe và bánh xe, trong khi `mobile_sam.pt` làm vỡ mask thành **2 polygon rời rạc** (`245,541 px`, `35.2%` ảnh) và làm khuyết mất mảng mui xe phía trước. SAM 3 (`facebook/sam3`, ~3.5 GB) là model gated cần xin quyền HuggingFace và đòi hỏi GPU mạnh nên không phù hợp với máy chỉ có CPU.
- **Lỗi nào không sửa được bằng đổi ngưỡng? Sửa thế nào?**
  - Lỗi **mù lớp ngoài tập huấn luyện (out-of-vocabulary / ontology mismatch)**: YOLO26 chỉ biết 80 lớp COCO nên luôn gán nhầm tuk-tuk là `truck` (`img_011.jpg`, `conf=0.77`) hoặc `car` (`img_013.jpg`, `conf=0.42`), và cắt phần bánh sau của xích lô thành `bicycle` (`img_032.jpg`, `conf=0.70`). Dù nâng hay hạ ngưỡng `conf`, YOLO26 không bao giờ sinh ra nhãn `three_wheeler`.
  - **Cách sửa:** Kết hợp detector open-vocabulary **YOLOE** với prompt văn bản (`tuk-tuk`, `cyclo`, `cargo tricycle`), sau đó dùng `merge_prelabels.py` đặt `yoloe_all` làm nguồn ưu tiên số 1 để đè lên các box `car`/`truck`/`bicycle` sai lớp của YOLO26 tại vùng chồng lấp ($\text{IoU} \ge 0.6$).
- **Box điểm cao mà sai hoặc điểm thấp mà đúng (ảnh nào):**
  - **Điểm cao mà sai:** Trong `img_011.jpg` (`work/prelabel/t_y26/viz/img_011.jpg`), chiếc tuk-tuk ở sát mép trái bị YOLO26 gán là `truck` với điểm tin cậy rất cao `conf = 0.77`; trong `img_032.jpg`, phần bánh sau của chiếc xích lô bị YOLO26 gán là `bicycle` với `conf = 0.70`.
  - **Điểm thấp mà đúng:** Trong `img_032.jpg` (`work/prelabel/t_yoloe/viz/img_032.jpg`), chiếc xích lô đỗ bên phải được YOLOE phát hiện đúng lớp `three_wheeler` nhưng chỉ có `conf = 0.27` (và ở `img_002.jpg`, chiếc xe ba gác chở hàng bên trái có `conf = 0.40`). Nếu đặt ngưỡng chung `0.50` thì cả hai chiếc xe ba bánh thật này đều bị xóa mất.
- **Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:**
  - Tập gold chỉ có **4 ảnh (`44` đối tượng)**; việc vừa dùng 4 ảnh này để chọn ngưỡng (`threshold_sweep.py`) vừa dùng chính 4 ảnh đó để báo cáo Precision/Recall của máy ở Part 4 là đánh giá **in-sample (chọn và chấm trên cùng dữ liệu)**, dẫn đến số đo bị thiên lệch lạc quan (overfitting ngưỡng vào 4 ảnh). Ngoài ra, các lớp ít đối tượng (`bicycle=3`, `truck=1`, `bus=0`) hoàn toàn không đủ cỡ mẫu thống kê để ước lượng đường cong Precision–Recall.

## 2. AI error profile (từ `qc/review_stats.md`)

> Nguồn: `submission/qc/review_stats.md` và `submission/qc/review_stats.json` (tính trên 24 ảnh batch sau khi đã loại **28 đối tượng nằm quanh các vùng bẫy nhân tạo** để không làm méo chất lượng thực của model; ngưỡng `accepted`: $\text{IoU} \ge 0.9$).

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---:|---:|---|
| `accepted` | 363 | `363/435 = 83.4%` | `435` box máy (ngoài vùng bẫy) |
| `edited` | 21 | `21/435 = 4.8%` | `435` box máy (ngoài vùng bẫy) |
| `deleted` | 51 | `51/435 = 11.7%` | `435` box máy (ngoài vùng bẫy) |
| `added` | 55 | `55/439 = 12.5%` | `439` đối tượng sau duyệt (ngoài vùng bẫy) |

- Trong số `51 deleted` + `55 added`, có **6 trường hợp là đổi lớp** (vị trí box của máy đúng nhưng gán sai lớp nên hợp đồng chấm tính là 1 `deleted` + 1 `added`). Do đó số đối tượng mà máy **bỏ sót thật sự** là $55 - 6 = 49/439 = 11.2\%$ đối tượng sau duyệt.
- **Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?**
  - Lớp tệ nhất là **`three_wheeler`**: trong `43` box máy dự đoán (`11 + 8 + 24`), chỉ có `11/43 = 25.6%` được giữ nguyên (`accepted`), phải chỉnh sửa kích thước `8/43 = 18.6%` (`edited`), phải xóa bỏ tới **`24/43 = 55.8%` (`deleted`)**, và phải vẽ thêm `6` đối tượng (`added`).
  - **Nguyên nhân:** Để đạt recall 100% trên gold ở Part 3, ngưỡng của `three_wheeler` được hạ xuống khá thấp (`0.15`). Ở ngưỡng `0.15`, YOLOE sinh ra nhiều box false positive đè lên đuôi xe tải, xe buýt hoặc quầy hàng ven đường (ví dụ ở `img_003.jpg` box `#17 conf=0.25`, `img_013.jpg` box `#109`), đồng thời box của YOLOE trên xích lô/xe ba gác thường chỉ bao phần đầu/ghế khách mà thiếu phần khung sau nên phải kéo lại (`edited = 8`).
- **So với gold: precision / recall của máy:**
  - Trên 4 ảnh gold (mẫu số của recall: **44 đối tượng gold**; mẫu số của precision: **61 box máy dự đoán** ngoài vùng ignore):
    - **Precision của máy:** `39/61 = 63.9%` (KTC 95%: `51%–75%`, có `TP = 39`, `FP = 22`).
    - **Recall của máy:** `39/44 = 88.6%` (KTC 95%: `76%–95%`, có `TP = 39`, `FN = 5`).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy:** Bắt được **`8/12 = 66.7%`** (KTC 95%: `39%–86%`, thấp hơn ngưỡng mục tiêu `10/12 = 80.0%`). Cụ thể theo từng loại (`submission/qc/review_stats.json`):
  - `extra` (box ma): bắt **`3/3 = 100%`** (`img_004.jpg`, `img_010.jpg`, `img_017.jpg`).
  - `loose` (box lỏng/lệch): bắt **`2/3 = 66.7%`** (bắt được ở `img_005.jpg`, `img_031.jpg`; **lọt 1 bẫy ở `img_034.jpg`** trên box `motorbike #464`).
  - `missing` (xoá vật thật): bắt **`2/3 = 66.7%`** (bắt được ở `img_026.jpg`, `img_033.jpg`; **lọt 1 bẫy ở `img_006.jpg`** trên 1 `person` nhỏ `[489, 315, 24, 30]` ở giữa hẻm).
  - `wrong_class` (đổi sai lớp): bắt **`1/3 = 33.3%`** (bắt được ở `img_027.jpg`; **lọt 2 bẫy ở `img_009.jpg`** khi `person` bị đổi thành `bicycle` và **`img_013.jpg`** khi chiếc xe van trắng `car` ở góc trên bên phải bị đổi thành `truck`).
- **Vì sao lọt (bạn đã bỏ bước nào của checklist)?**
  - **Lọt `wrong_class` (2/3 bẫy lọt ở `img_009`, `img_013`):** Khi nhìn vào ảnh có nhiều đối tượng hoặc có một vật thể lớn ở tiền cảnh (như chiếc tuk-tuk lớn chiếm gần nửa ảnh ở `img_013.jpg`), mắt bị hút vào việc sửa vật chính và thấy các vật ở hậu cảnh (như chiếc xe van trắng `#111` ở góc xa `img_013`) "đã có box ôm khít rồi" nên **bỏ qua Bước 3 (Kiểm tra Sai nhãn)**, không đọc kỹ tên nhãn `truck` vs `car` hay `bicycle` vs `person`.
  - **Lọt `missing` (`img_006.jpg`):** Bỏ qua hoặc làm quá nhanh **Bước 0 (Ẩn nhãn máy, quét ảnh đếm vật trước)** ở vùng trung tâm hẻm sâu có nhiều người đứng chồng lên nhau, dẫn đến thiên kiến "neo nhãn" (chỉ nhìn quanh những chỗ máy đã vẽ box).
  - **Lọt `loose` (`img_034.jpg`):** Ở ảnh cuối lô (`img_034` là ảnh đông phương tiện lúc chiều tối), do mệt mỏi nên chấp nhận box `motorbike` bị dịch lệch nhẹ mà không kéo sát lại ở **Bước 4 (Kiểm tra Lỏng)** để đạt $\text{IoU} \ge 0.9$ với box gốc.
- **Accepted cao (`83.4%`) mà bắt bẫy thấp (`66.7%`) nói gì về cách bạn duyệt?**
  - Đây là biểu hiện điển hình của **thiên kiến tự động hóa (automation bias)**: tỉ lệ `accepted = 363/435 = 83.4%` cao và tốc độ duyệt nhanh (`45 phút / 24 ảnh = 1.88 phút/ảnh`) một phần đến từ việc người duyệt **tin tưởng quá mức vào các box có sẵn của máy** (đặc biệt là khi biên box trông hợp lý), nên chỉ tập trung xóa các box "ma" lộ liễu (`extra` bắt 3/3) mà lướt qua lỗi sai nhãn (`wrong_class` chỉ bắt 1/3) và lỗi bỏ sót vật nhỏ ở xa (`missing`). Đồng thời, tỉ lệ `added = 12.5%` thấp không chứng minh máy ít bỏ sót nếu chính người duyệt cũng cùng bỏ sót với máy (như bẫy `missing` ở `img_006.jpg`).

## 4. Máy có đáng dùng không?

- **Thời gian (từ `submission/time_log.csv`):**
  - Gán tay từ đầu (`gold_manual`): `20 phút / 4 ảnh = 5.00 phút/ảnh` (`44` đối tượng $\approx 27.3\text{ giây/đối tượng}$).
  - Duyệt có pre-label (`batch_review`): `45 phút / 24 ảnh = 1.88 phút/ảnh` (`453` đối tượng $\approx 6.0\text{ giây/đối tượng}$), **tiết kiệm `62.5%` thời gian trên mỗi ảnh** (nhanh hơn `2.67 lần`).
  - **Thiên lệch khi so sánh thời gian:** Khi gán 4 ảnh gold đầu tiên, mình vừa phải đọc guideline v1.0 vừa làm quen công cụ CVAT; đến khi duyệt 24 ảnh batch thì đã thuộc phím tắt và quy tắc nên tốc độ tự nhiên nhanh hơn. Hơn nữa, việc duyệt nhanh `1.88 phút/ảnh` đi kèm cái giá là để lọt `4/12` bẫy (nếu duyệt kỹ hơn để đạt $\ge 80\%$ bẫy thì thời gian thực tế sẽ khoảng `2.5–3.0 phút/ảnh`).
- **Kết luận:** **Có đáng dùng cho 6 lớp COCO (đặc biệt là `person`, `motorbike`, `car`), nhưng cần siết lại ngưỡng và quy trình duyệt cho `three_wheeler`.**
  - **Dựa trên số liệu:** Với `person` (`accepted = 179/187 = 95.7%` box máy), `motorbike` (`117/135 = 86.7%`), và `car` (`46/54 = 85.2%`), máy giúp giảm hơn một nửa thời gian vẽ box. Ngược lại, với `three_wheeler`, tỉ lệ `deleted` lên tới `24/43 = 55.8%` và `edited` là `8/43 = 18.6%` (`accepted` chỉ `25.6%`), nên việc phải xóa 24 box thừa của YOLOE ở ngưỡng `0.15` gây tốn công và dễ sót box thừa (như phát hiện ở bước QC).
- **Ba quy tắc giảm thiên kiến của bạn (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):**
  1. **Quy tắc 1 (Chống bỏ sót / `missing`):** Bắt buộc bấm phím `H` trên CVAT để **ẩn toàn bộ nhãn máy trong 15 giây đầu tiên** trên 100% ảnh, đếm nhẩm số lượng `person` và `three_wheeler` ở từng góc phần tư rồi mới bật nhãn lên (kiểm chứng: tỉ lệ bắt bẫy `missing` phải đạt `3/3 = 100%`).
  2. **Quy tắc 2 (Chống sai lớp / `wrong_class` & box nhỏ `< 20px`):** Dùng tính năng **Filter** của CVAT lọc riêng từng lớp dễ nhầm (`truck`, `bicycle`, `three_wheeler`) sau khi duyệt xong mỗi 8 ảnh để kiểm tra chéo 100% nhãn lớp của từng box, và chạy `validate_labels.py` để xóa sạch 100% các box máy cao `< 20 px` (kiểm chứng: `validate_labels.py` đạt `0 cảnh báo` box `< 20 px` và bắt bẫy `wrong_class` $\ge 2/3$).
  3. **Quy tắc 3 (Chống mệt mỏi & lướt nhanh):** Đặt **giới hạn thời gian sàn tối thiểu `2.5 phút/ảnh`** cho các ảnh đông ($> 20$ đối tượng) và **nghỉ bắt buộc 5 phút sau mỗi 8 ảnh** (ghi dòng `break` vào `time_log.csv` để kiểm chứng).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---:|---|---|---|
| `img_013.jpg` (xe tuk-tuk tiền cảnh) | **SAM 2.1** (`sam2.1_b.pt`) | **3 click** (`2 pos`: `400,480`, `600,110`; `1 neg`: `720,300`) | **Có (một phần):** 1 click âm tại lưng áo tài xế (`720,300`) chỉ khoét được vùng lưng áo xanh, nhưng mask xe (`317,255 px`, `45.4%` ảnh, `1 polygon`) vẫn tràn sang đầu, quần tài xế và hành khách mặc áo hồng ngồi trong. | Ranh giới giữa khoang xe hở và người ngồi bên trong có cùng độ sâu/tương phản nên chỉ 1 click âm không đủ tách toàn bộ cơ thể tài xế khỏi khung xe. | `submission/masks/tuktuk_img013_sam21.jpg` |
| `img_013.jpg` (cùng xe tuk-tuk, đối chiếu SAM 1) | **SAM 1** (`mobile_sam.pt`) | **3 click** (`2 pos`: `400,480`, `600,110`; `1 neg`: `720,300`) | **Có (ít hơn ở lưng/quần nhưng vẫn tràn vào mặt và bàn chân tài xế).** | Diện tích mask giảm còn `245,541 px` (`35.2%` ảnh) nhưng bị **vỡ thành 2 polygon rời rạc**, làm khuyết mất mảng mui bạt phía trước đầu tài xế và viền bánh xe răng cưa hơn `sam2.1_b.pt`. | `submission/masks/tuktuk_img013_sam1.jpg` |
| `img_032.jpg` (xích lô tiền cảnh) | **SAM 2.1** (`sam2.1_b.pt`) | **3 click** (`2 pos`: `410,360`, `430,270`; `1 neg`: `570,320`) | **Không:** 1 click âm (`570,320`) vào người đàn ông mặc áo khoác đen đứng cạnh đã loại bỏ hoàn toàn người này khỏi mask xích lô (`50,564 px`, `8.6%` ảnh, `1 polygon`). | Mask ôm rất chuẩn khoang ghế hoa văn và 2 bánh trước của xích lô, nhưng phần bánh/khung xe đạp phía sau bị người áo đen che khuất bị tách rời nên chưa nối vào cùng polygon. | `submission/masks/cyclo_img032_sam21.jpg` |
| `img_004.jpg` (xe tuk-tuk tiền cảnh bên phải) | **SAM 2.1** (`sam2.1_b.pt`) | **3 click** (`2 pos`: `620,370`, `560,260`; `1 neg`: `470,330`) | **Không gồm người đứng cạnh:** 1 click âm (`470,330`) tách sạch người đàn ông đứng sát bên trái xe; mask (`91,631 px`, `13.1%` ảnh, `1 polygon`) bao trọn cả kính chắn gió và thân xe. | Phần hộp đèn chữ "TAXI" màu trắng nhỏ trên đỉnh mui có độ tương phản cao với mui bạt tối màu nên chưa được gộp vào mask nếu không chấm thêm 1 click dương lên hộp đèn. | `submission/masks/tuktuk_img004_sam21.jpg` |

- **Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?**
  - **Quy tắc giúp ích:** Mục 4.2.2 (*"không vẽ lỗ nếu lỗ nhỏ hơn ~5% diện tích vật"*) và Mục 4.2.3 (*"Bóng đổ không thuộc vật. Kính xe thuộc vật"*) giúp xác định ngay rằng mask 1 polygon đặc của `sam2.1_b.pt` bao gồm cả kính chắn gió (`img_004`) và các nan sắt nhỏ quanh thành xe (`img_013`) là hoàn toàn hợp lệ, không cần khoét từng ô thoáng nhỏ.
  - **Quy tắc còn thiếu (đã bổ sung vào v1.1 `CAL-04`, `CAL-05`):** Guideline v1.0 mục 4.2 chưa nói rõ cách xử lý khi **người lái/hành khách ngồi lọt trong lòng khung xe hở** (như tài xế tuk-tuk ở `img_013` chia xe thành vùng trước và vùng sau): nếu loại trừ người lái bằng nhiều click âm thì mask của chiếc xe có được phép tách thành **nhiều polygon rời nhau (multipolygon)** hay không. Trong `guideline_v1.1_calibrated.md`, mình đã chốt: phải dùng click âm loại trừ phần cơ thể người lộ trực tiếp ở hông xe hở và chấp nhận multipolygon khi vật bị người phía trước cắt ngang (như xích lô `img_032`).

## 6. Calibration (Part 6)

- **Số quyết định:** **6 quyết định** (`CAL-01` đến `CAL-06` trong `submission/qc/calibration_decisions.csv`, gồm 4 ca `cau_hoi_mo`, 1 ca `loi_may`, 1 ca `bay_lot`).
- **Quyết định quan trọng nhất:**
  - **`CAL-05` (`nguon = loi_may`, `img_002|img_013|img_032`):** Xác định rõ ranh giới hình học và nhãn lớp cho `three_wheeler` (tuk-tuk, xích lô, ba gác) trước lỗi hệ thống của cả 3 model (YOLO26 gán nhầm thành `car`/`truck`, YOLOE chỉ đóng khung khoang ghế trước của xích lô mà bỏ bánh xe đạp sau, SAM 2.1 nhận box cụt của YOLO26 nên cắt mất đuôi xe).
  - **`CAL-02` (`nguon = cau_hoi_mo`, `img_005|img_018|img_024`):** Chốt quy tắc xử lý các box máy có chiều cao $< 20\text{ px}$ ở hậu cảnh (phải xóa bỏ các box lẻ $< 20\text{ px}$ của máy và gom cụm $\ge 5$ xe nhỏ vào 1 `ignore_region`), giải quyết trực tiếp 9 cảnh báo của `validate_labels.py`.
- **Quy tắc mới đã vào `submission/guidelines/guideline_v1.1_calibrated.md`:**
  1. Phân biệt xe đạp điện (`bicycle` — có bàn đạp + khung mảnh) và xe máy điện (`motorbike` — sàn để chân liền khối dáng scooter) (`CAL-01`, `img_018`, `img_024`).
  2. Bắt buộc xóa các box máy lẻ cao $< 20\text{ px}$ và gom cụm $\ge 5$ xe nhỏ/che khuất $> 80\%$ thành 1 `ignore_region` (`CAL-02`, `img_005`, `img_018`, `img_024`).
  3. Loại trừ hàng hóa rời nhô ra $> 20\%$ chiều dài xe khỏi box/mask của xe và không gộp quang gánh/thúng hàng vào box `person` (`CAL-03`, `img_002`, `img_014`).
  4. Quy đổi ngưỡng nhìn thấy $\ge 20\%$ cho người ngồi trong tuk-tuk thành "thấy rõ tối thiểu trọn phần đầu + hai vai"; kính chắn gió thuộc mask xe nhưng phải tách riêng người lộ qua hông xe hở (`CAL-04`, `img_004`, `img_013`).
  5. Box/mask của xích lô và ba gác (`three_wheeler`) phải ôm trọn cả khoang khách/thùng hàng trước lẫn khung xe đạp/mô tô gắn liền phía sau (`CAL-05`, `img_002`, `img_013`, `img_032`).
- **Nguyên nhân gốc (phân biệt rõ 3 nhóm):**
  - **Do guideline v1.0 cố ý để mở (`CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`):** Cần viết thành quy tắc mới trong `guideline_v1.1_calibrated.md`.
  - **Do hạn chế của model pre-label (`CAL-05`):** YOLO26 closed-set không biết `three_wheeler` và YOLOE ở ngưỡng `0.15` sinh nhiều box thừa (`deleted = 24/43`). Cần sửa ở recipe chọn ngưỡng và bước review.
  - **Do lỗi thao tác của người duyệt vì thiên kiến tự động hóa (`CAL-06`):** Để lọt 4/12 bẫy (`img_006`, `img_009`, `img_013`, `img_034`) và sót 9 box máy $< 20\text{ px}$. Đây là lỗi thực thi của con người, không phải do guideline mơ hồ.

## 7. QC và quyết định (Part 6)

- **Mẫu:** **`6/24` ảnh batch** (`1` ảnh ca khó có tick `can_review` là `img_022.jpg` + `5` ảnh ngẫu nhiên với `seed = 42` là `img_003.jpg`, `img_006.jpg`, `img_013.jpg`, `img_014.jpg`, `img_033.jpg`), tổng cộng **`95` đối tượng** (`submission/qc/qc_sample.csv`). Người kiểm: chính học viên (**hạn chế:** người duyệt tự kiểm lại nhãn của mình nên dễ bị mù lỗi lặp lại so với kiểm định chéo độc lập).
- **Hai cách đếm tỉ lệ lỗi kèm mẫu số (từ `submission/qc/qc_report_auto.md`):**
  - Lỗi theo **đối tượng**: **`6/95 = 6.3%`** (mẫu số: `95` đối tượng trong 6 ảnh mẫu).
  - Lỗi theo **ảnh**: **`5/6 = 83.3%`** (mẫu số: `6` ảnh trong mẫu; ngưỡng Release yêu cầu $\le 2\%$).
- **Top lỗi (trong 6 lỗi phát hiện trên mẫu QC) + ảnh ví dụ:**
  1. **`thua` (`3/6 = 50.0%` tổng số lỗi — lỗi hệ thống):**
     - `img_003.jpg` (đối tượng `#17`): box `three_wheeler` (`conf=0.25`) của YOLOE nằm đè lên đoàn xe máy/người ở xa bên trái chưa được xóa.
     - `img_013.jpg` (đối tượng `#109`): box `three_wheeler` của YOLOE bắt nhầm phần đuôi xe ở góc trên bên phải.
     - `img_033.jpg` (đối tượng `#396`): box `motorbike` ở xa chỉ cao `18.6 px` ($< 20\text{ px}$ theo quy tắc 4.1.6 không gán lẻ).
  2. **`bo_sot` (`1/6 = 16.7%`):** `img_006.jpg` — bỏ sót `1` đối tượng `person` đi bộ ở giữa hẻm sâu (trùng với bẫy `missing`).
  3. **`sai_lop` (`1/6 = 16.7%`):** `img_013.jpg` (đối tượng `#111`) — chiếc xe van trắng ở góc trên bên phải bị giữ nguyên nhãn bẫy `truck` thay vì đổi về `car`.
  4. **`sai_vi_tri_kich_thuoc` (`1/6 = 16.7%`):** `img_014.jpg` (đối tượng `#126`) — box `person` của người gánh hàng rong bao trùm cả đòn gánh và hai thúng hàng hai bên thay vì chỉ ôm sát cơ thể người (vi phạm `CAL-03`).
- **Giới hạn mẫu nhỏ (cận trên 95% trong `qc_report_auto.md`):**
  - Với `5/6` ảnh có lỗi, cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thật lên tới **`97%`**.
  - Quan trọng hơn, về mặt thống kê, ngay cả khi mẫu `6` ảnh có **`0/6 = 0.0%` lỗi**, thì theo quy tắc $3/n$ (hoặc Wilson score), cận trên 95% của tỉ lệ lỗi thật vẫn ở mức **`~39%–50%`**, hoàn toàn **không thể chứng minh được lô dữ liệu đạt ngưỡng $\le 2\%$**. Muốn chứng minh tỉ lệ lỗi $\le 2\%$ với độ tin cậy 95% khi có 0 lỗi, cỡ mẫu QC tối thiểu phải là $n \approx 3 / 0.02 = 150\text{ ảnh}$ (lớn hơn cả tổng số 24 ảnh của lô hiện tại).
- **Script kiểm được / không kiểm được:**
  - **Kiểm được tự động (`validate_labels.py`, `review_stats.py`, `qc_report.py`):** Đủ số lượng ảnh, tên lớp thuộc `taxonomy.json`, tọa độ box không tràn khỏi khung ảnh, cảnh báo box có chiều cao $< 20\text{ px}$, tính IoU đối chiếu với pre-label/gold/bẫy đã cài sẵn.
  - **Không kiểm được tự động (cần mắt người đối chiếu ảnh gốc):** Box có thực sự ôm sát đối tượng với sai số $\le 2\text{ px}$ hay không (như `#126` ở `img_014.jpg` ôm luôn gánh hàng), nhãn lớp có đúng với vật thể thực tế ngoài các vị trí bẫy hay không (như `#17` ở `img_003.jpg`), những vật mà **cả máy lẫn người duyệt cùng bỏ sót** trên 24 ảnh batch, và người duyệt có thực sự ẩn nhãn quét ảnh trước theo checklist hay không.

**Quyết định: Rework**, ngày `2026-11-18`.
- **Lý do (đi theo cây quyết định Hold $\rightarrow$ Rework $\rightarrow$ Release):**
  1. **Kiểm tra Hold:** Dữ liệu ảnh rõ nét, không bị hỏng, và cả 7 câu hỏi mở của guideline v1.0 đã được giải quyết dứt điểm trong `guideline_v1.1_calibrated.md` (`--guideline-gap 0`, `--data-problem 0`) $\rightarrow$ **Không Hold**.
  2. **Kiểm tra Rework:** Lô nhãn vi phạm cả 3 tiêu chí chất lượng nên **bắt buộc phải Rework** (không thể Release):
     - Tỉ lệ ảnh có lỗi trong mẫu QC là **`5/6 = 83.3%`** (vượt xa ngưỡng tối đa `2%`).
     - Tồn tại **lỗi hệ thống `'thua'`** chiếm **`3/6 = 50.0%`** số lỗi (do giữ lại các box false positive điểm thấp của YOLOE và các box máy cao $< 20\text{ px}$).
     - Tỉ lệ bắt bẫy chỉ đạt **`8/12 = 66.7%`** (thấp hơn ngưỡng tối thiểu `80%` = `10/12`).
- **Nếu Rework / Hold, việc cần làm tiếp:**
  1. Nâng ngưỡng lọc `three_wheeler` của YOLOE từ `0.15` lên `0.25`–`0.30` khi merge để giảm số box thừa (`deleted` hiện đang là `24/43`).
  2. Mở lại Task `batch-review` trên CVAT, áp dụng 3 quy tắc giảm thiên kiến ở Mục 4 và `guideline_v1.1_calibrated.md` để: (a) xóa toàn bộ 9 box nhỏ `< 20 px` ở `img_018, img_027, img_033, img_034` được `validate_labels.py` cảnh báo, (b) xóa các box `three_wheeler` thừa ở hậu cảnh (`img_003 #17`, `img_013 #109`), (c) co lại box người gánh hàng rong `img_014 #126`, và (d) sửa 4 vị trí bẫy bị lọt (`img_006`, `img_009`, `img_013`, `img_034`).
  3. Xuất bản sửa ra `submission/labels/qc_fixed.json`, chạy lại `build_release.py`, `qc_sample.py --force` và `qc_report.py` để nghiệm thu vòng 2.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài (công cụ nào, việc gì, kiểm tra thế nào):**
  - Sử dụng trợ lý **Antigravity (Gemini)** trong IDE để hỗ trợ: (1) cài đặt môi trường `.venv`, tải trọng số model và khắc phục lỗi xung đột 2 bộ container Docker CVAT (`cvat` và `cvat281` trên cổng `8081`), (2) chạy chuỗi lệnh CLI của các script (`prelabel.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `sam2_points.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`), và (3) hỗ trợ định dạng bảng biểu Markdown trong báo cáo.
  - **Cách kiểm tra:** Toàn bộ việc gán nhãn 4 ảnh gold (`20 phút`) và duyệt 24 ảnh batch (`45 phút`) được mình thực hiện trực tiếp bằng tay trên giao diện CVAT (`http://localhost:8081`); mọi con số trong báo cáo và nhận xét lỗi đều được đối chiếu trực tiếp với các file đầu ra trong `submission/qc/*` và ảnh trực quan trong `work/prelabel/*/viz/`, `submission/masks/`, `work/qc/review/`.
- **Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?**
  - **Không.** Tập gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) được mình tự gán tay hoàn toàn trên CVAT và xuất ra `submission/labels/gold_manual.json` trước khi chạy bất kỳ lệnh `prelabel.py` nào trên tập dữ liệu.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:**
  - **Máy:** Laptop HUAWEI chạy Windows 11, Python 3.10.8, **chỉ có CPU** (không có GPU rời NVIDIA). Thời gian chạy `yolo26m` mất khoảng `5.8s–11.8s/ảnh`, `yoloe-26s-seg` mất `2.6s/ảnh`, và `sam2.1_b.pt` mất `45.2s/ảnh`.
  - **CVAT:** Docker Desktop + CVAT `v2.81.0` chạy tại `http://localhost:8081`.
  - **Vướng mắc cài đặt & cách xử lý:** Ban đầu máy chạy song song 2 bộ container (`cvat_*` từ buổi học trước và `cvat281_*` mới, tổng cộng 35 containers) khiến `cvat281_server` quá tải trả về lỗi `502/504 Gateway Timeout`, đồng thời database mới của `cvat281` chưa có tài khoản superuser. Đã khắc phục bằng cách tắt bộ container `cvat` cũ và khởi tạo tài khoản superuser trong `cvat281_server`.
- **Dùng cache `data/cache/` thay vì tự chạy?**
  - **Không.** Đã tự chạy trực tiếp toàn bộ 7 lượt model (`r01`–`r07` trong `work/runs.csv` và `submission/model_log.csv`) bằng CPU trên máy cá nhân.
- **Lỗi / bug script đã gặp (nếu có):**
  - Không gặp bug trong mã nguồn của các script trong `scripts/`; chỉ gặp lỗi quyền ghi thư mục cấu hình mặc định `AppData\Roaming\Ultralytics` khi chạy trong môi trường hạn chế quyền và khắc phục dễ dàng bằng biến môi trường `YOLO_CONFIG_DIR`.
- **Một điều bạn làm khác lần sau:**
  - Lần sau khi duyệt pre-label trên CVAT, mình sẽ **luôn bấm phím `H` ẩn nhãn máy để quét ảnh 15 giây trước** và **dùng bộ lọc lớp (Filter) để rà soát riêng từng lớp phương tiện (`truck`, `car`, `three_wheeler`)** cũng như xóa sạch các box máy $< 20\text{ px}$ trước khi export, thay vì duyệt lướt theo các box có sẵn làm lọt mất các bẫy `wrong_class` và box thừa ở hậu cảnh.

