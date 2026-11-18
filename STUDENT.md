# Thông tin học viên

- Họ và tên: Phạm Võ Quang Hưng
- MSHV: 2A202603259
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: QuangHung259
- Repo bài nộp (URL): https://github.com/QuangHung259/K5-L23-Day08-PhamVoQuangHung-2A202603259-AI-Assisted-Labeling
- Commit chốt bài (hash): ea422ad
- Máy: Windows 11, CPU (không có GPU)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 363/435 = 83.4% / 21/435 = 4.8% / 51/435 = 11.7% (mẫu số: 435 box máy ngoài vùng bẫy); added: 55/439 = 12.5% (mẫu số: 439 đối tượng sau duyệt ngoài vùng bẫy; 6 ca đổi lớp)
- So với gold: precision / recall của **máy**: 39/61 = 63.9% (mẫu số: 61 box máy) / 39/44 = 88.6% (mẫu số: 44 đối tượng gold)
- Bắt bẫy: 8/12 = 66.7% (extra 3/3, loose 2/3, missing 2/3, wrong_class 1/3)
- Thời gian: gán tay 5.00 phút/ảnh (20 phút / 4 ảnh); duyệt có pre-label 1.88 phút/ảnh (45 phút / 24 ảnh)
- Lỗi QC: 83.3% theo ảnh (5/6 ảnh mẫu), 6.3% theo đối tượng (6/95 đối tượng trong mẫu)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh lỗi trong mẫu QC cao (5/6 = 83.3% > 2%), có lỗi hệ thống 'thua' (3/6 lỗi do box thừa three_wheeler và box < 20px) và tỉ lệ bắt bẫy đạt 8/12 = 66.7% (< 80%).
