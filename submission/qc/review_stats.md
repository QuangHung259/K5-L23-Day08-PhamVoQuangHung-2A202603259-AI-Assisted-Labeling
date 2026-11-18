# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 435; đối tượng sau duyệt: 439; ngưỡng accepted: IoU >= 0.9
- Đã **loại 28 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 363 | 363/435 = 83.4% | 435 box máy |
| edited (sửa) | 21 | 21/435 = 4.8% | 435 box máy |
| deleted (xoá) | 51 | 51/435 = 11.7% | 435 box máy |
| added (vẽ thêm) | 55 | 55/439 = 12.5% | 439 đối tượng sau duyệt |

Trong số deleted + added, **6 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 49.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 179 | 4 | 4 | 18 |
| bicycle | 6 | 0 | 2 | 4 |
| motorbike | 117 | 9 | 9 | 15 |
| car | 46 | 0 | 8 | 8 |
| bus | 3 | 0 | 0 | 3 |
| truck | 1 | 0 | 4 | 1 |
| three_wheeler | 11 | 8 | 24 | 6 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 44 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 39 | 22 | 5 | 63.9% (KTC 95%: 51–75%) | 88.6% (KTC 95%: 76–95%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 8/12 = 66.7% (KTC 95%: 39–86%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 1/3
  - loose: 2/3
  - missing: 2/3
  - extra: 3/3
  - lọt: img_013.jpg (wrong_class), img_034.jpg (loose), img_006.jpg (missing), img_009.jpg (wrong_class)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
