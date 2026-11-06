# Hạn chế của kiến trúc UniFi

UniFi [1] nhận dạng cử chỉ từ CSI Wi-Fi bằng mạng đa góc nhìn. Mỗi bộ thu (view) có hai ảnh đầu vào: phổ Doppler (DFS) và pha của CSI ratio. Hai ảnh này được mã hoá riêng bằng ResNet18, rồi đi qua các khối xử lý giữa các view: masked cross-view attention, temporal-aware weighted average, multi-scale convolution và view-level attention. Phần này phân tích các hạn chế ở mức ý tưởng và kiến trúc. Phạm vi là bài toán nhận dạng cử chỉ của một người, xử lý offline.

## 1. Nguyên nhân gốc: ResNet18 trừu tượng hoá quá sớm

Trong UniFi, mỗi view được mã hoá độc lập và trọn vẹn bằng ResNet18 *trước khi* các view tương tác với nhau. ResNet18 được thiết kế cho ảnh tự nhiên: nhiều tầng tích chập và pooling, tổng stride 32. Vì vậy, khi đến các khối xử lý giữa các view, cả trục tần số lẫn trục thời gian đều đã bị nén thành các biểu diễn trừu tượng. Các hạn chế dưới đây đều là hệ quả của lựa chọn này.

### 1.1. Trục Doppler bị nén nên attention giữa các view kém hiệu quả

Sau ResNet18, trục tần số của ảnh DFS bị gộp lại. Mỗi view chỉ còn một chuỗi vector đặc trưng theo thời gian. Thông tin như "tại thời điểm *t*, view *i* thấy dịch Doppler 30 Hz, còn view *j* thấy 10 Hz" không còn ở dạng có thể so sánh trực tiếp. Mà về mặt vật lý, đây chính là thông tin cần thiết để kết hợp các view: DFS của mỗi link là hình chiếu của vector vận tốc lên một hướng khác nhau. Khối cross-view attention phía sau vì thế chỉ làm việc trên các đặc trưng đã mất cấu trúc vật lý.

Ablation của UniFi phản ánh điều này (Bảng 3 trong [1]):
- Bỏ cấu trúc đa view làm accuracy giảm **4.8%** ở cross-location và **9.6%** ở cross-orientation. Thông tin giữa các view rõ ràng rất có giá trị.
- Bỏ riêng khối attention chỉ làm giảm **0.4–1.5%**. Tác giả cũng thừa nhận việc bỏ khối này *"does not significantly decrease model performance"*.

Như vậy, khối attention chưa khai thác được nguồn thông tin mà cấu trúc đa view mang lại.

### 1.2. Phổ CSI được xử lý như ảnh tự nhiên

ResNet18 coi DFS và pha như ảnh RGB. Với ảnh tự nhiên, một đối tượng xuất hiện ở đâu trong khung hình cũng mang cùng ý nghĩa. Tích chập có tính dịch chuyển tương đương, và khi kết hợp với pooling thì mạng gần như bất biến với vị trí của đặc trưng. Điều này không đúng với DFS: vị trí theo trục tần số biểu diễn trực tiếp vận tốc của chuyển động, nên một mẫu nằm ở 20 Hz hay 40 Hz là hai chuyển động khác nhau về tốc độ. Meng et al. [2] cũng chỉ ra rằng hai trục của CSI có ngữ nghĩa khác với ảnh, nên việc tìm mẫu không gian bằng CNN theo cách xử lý ảnh *"can often be meaningless"*.

### 1.3. Trục thời gian bị nén nên các khối xử lý thời gian mất ý nghĩa

ResNet18 được thiết kế để nhận dạng *nội dung* của ảnh, không để giữ *vị trí* của nội dung đó. Mạng liên tục giảm độ phân giải không gian qua các tầng stride và pooling, đồng thời mở rộng vùng tiếp nhận để mỗi đặc trưng ở tầng cuối bao quát được phần lớn ảnh. Khi áp dụng cho ảnh CSI, trục ngang là trục thời gian, nên hai tính chất này trở thành hai hạn chế:
- **Độ phân giải thời gian còn rất thấp.** Cả cử chỉ chỉ còn được biểu diễn bằng một chuỗi rất ngắn, mỗi phần tử đại diện cho một khoảng dài của cử chỉ.
- **Các phần tử của chuỗi không còn tách biệt theo thời gian.** Mỗi phần tử tổng hợp thông tin từ một vùng rộng, có thể là gần như toàn bộ cử chỉ, nên các phần tử kề nhau chồng lấp mạnh về nội dung.

Trong khi đó, toàn bộ các khối xử lý thời gian của UniFi đều được thiết kế với giả định rằng chuỗi đặc trưng còn giữ được cấu trúc thời gian chi tiết:
- **Temporal-aware weighted average** được đề xuất để bù độ lệch thời gian nhỏ giữa các view, do tốc độ lấy mẫu khác nhau. Nhưng sau ResNet, mỗi phần tử đã bao quát một khoảng thời gian lớn hơn nhiều so với độ lệch cần bù, và đã tự trộn thông tin của các thời điểm lân cận. Việc lấy trung bình có trọng số giữa các phần tử kề nhau gần như không còn tác dụng.
- **Multi-scale convolution** nhằm bắt các nét cử chỉ (đường thẳng, cung) ở các thang thời gian khác nhau. Trên một chuỗi ngắn mà mỗi phần tử đã nhìn thấy phần lớn cử chỉ, các kernel nhỏ và lớn đều bao phủ gần như cùng một phạm vi, nên ý nghĩa "đa thang" không còn.
- **Cross-view attention** tính tại từng bước thời gian, nên chỉ so khớp các view ở độ phân giải thô, không đủ để tách các nét ngắn của cử chỉ.

Thứ tự thời gian *bên trong* cử chỉ vì thế bị làm mờ đáng kể. Meng et al. [2] cho thấy thông tin thứ tự rất quan trọng với các hoạt động đối xứng: mô hình dựa trên CNN chỉ đạt 40–47% với cặp "đứng lên / ngồi xuống", trong khi mô hình dùng tích chập nhân quả theo thời gian đạt 99–100%.

### 1.4. Chuẩn hoá kích thước làm mất thang thời gian thật

Mọi mẫu được đưa về cùng kích thước ảnh, bất kể cử chỉ kéo dài bao lâu. Việc này làm mất **thời lượng thật** của cử chỉ: một cử chỉ kéo dài 1 s và một cử chỉ kéo dài 3 s cho ra ảnh có cùng chiều rộng. Ở nhánh pha, việc co giãn trục thời gian còn làm méo **tốc độ biến thiên pha**. Thông tin vận tốc chỉ còn được giữ ở trục tần số của DFS, nhưng như mục 1.2 đã phân tích, trục này lại bị ResNet làm mờ. Kết quả là các cử chỉ khác nhau chủ yếu về nhịp độ khó được phân biệt.

## 2. Cơ chế chọn view cố định

Spatial-aware view mask chọn các view để attend dựa trên **hướng đặt cố định của thiết bị**, ưu tiên các cặp khác hướng nhau. Tác giả lập luận rằng theo mô hình vùng Fresnel, hai view lệch hướng nhiều thì bổ sung cho nhau nhiều. Tuy nhiên, mức độ hữu ích của một view còn phụ thuộc vào **vị trí và hướng của người** so với thiết bị:
- Các bộ thu đặt gần nhau vẫn bổ sung cho nhau. Chúng quan sát cùng một chuyển động qua các kênh có nhiễu độc lập, nên giúp lọc nhiễu và tạo đồng thuận.
- Khi cử chỉ diễn ra gần các bộ thu này, phản xạ từ cơ thể tới chúng mạnh hơn so với thành phần truyền thẳng. Chúng có thể là những view mang thông tin rõ nhất.

Mask tĩnh vì vậy bỏ qua đúng các yếu tố domain mà UniFi hướng tới tổng quát hoá: vị trí và hướng của người. Bài báo cũng không có ablation riêng cho mask, nên giả định này chưa được kiểm chứng.

## 3. Tóm tắt

| Hạn chế | Hệ quả | Bằng chứng |
|---|---|---|
| ResNet18 nén trục Doppler trước khi các view tương tác | Attention giữa các view chỉ làm việc trên đặc trưng trừu tượng | Bỏ đa view: −4.8–9.6%; bỏ attention: chỉ −0.4–1.5% [1] |
| Xử lý DFS và pha như ảnh tự nhiên | Bỏ qua ý nghĩa vật lý của trục tần số (vận tốc) | Phân tích của Meng et al. [2] |
| ResNet18 nén trục thời gian thành chuỗi ngắn với các phần tử chồng lấp mạnh | Temporal-aware average, multi-scale conv và cross-view attention mất ý nghĩa; thứ tự trong cử chỉ bị làm mờ | CNN chỉ đạt 40–47% với cặp hoạt động đối xứng [2] |
| Chuẩn hoá kích thước ảnh | Mất thời lượng thật, méo tốc độ biến thiên pha | Phân tích |
| Mask chọn view cố định theo vị trí thiết bị | Bỏ qua vị trí và hướng của người; loại các view gần nhau dù có thể giàu thông tin nhất | Không có ablation riêng trong [1] |

Các hạn chế này gợi ý ba hướng cải tiến: (i) thay ResNet18 bằng bộ mã hoá giữ được cấu trúc thời gian, ví dụ mạng tích chập thời gian (TCN) như trong [2]; (ii) cho các view tương tác sớm hơn, khi trục Doppler chưa bị nén; (iii) chọn view thích nghi theo từng mẫu thay cho mask cố định.

## Tài liệu tham khảo

[1] Y. Liu et al., "UniFi: A Unified Framework for Generalizable Gesture Recognition with Wi-Fi Signals Using Consistency-guided Multi-View Networks," *Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.*, vol. 7, no. 4, 2023. DOI: 10.1145/3631429.

[2] W. Meng, Z. Liu, B. Li, W. Cui, J. T. Zhou, and L. Zhang, "GrapHAR: A Lightweight Human Activity Recognition Model by Exploring the Sub-Carrier Correlations," *IEEE Trans. Wireless Commun.*, vol. 23, no. 4, pp. 2755–2770, 2024.
