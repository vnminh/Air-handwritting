Có. Tôi đã đọc báo cáo đính kèm và đối chiếu với scope chính thức của SOICT 2026. Theo tôi, đề tài này có nền tảng đủ tốt để phát triển thành một bài SOICT 2026, đặc biệt phù hợp với track Multimedia Processing, Computer Vision, and Multimodal Intelligence hoặc Applied AI. Air-writing, pattern recognition, AR/VR và human-centered interaction đều nằm khá sát scope của hội nghị. SOICT 2026 xuất bản Springer CCIS và proceedings dự kiến được index Scopus/EI; hạn full paper hiện là 16/09/2026, tối đa 12 trang không tính references theo trang submission.

Tuy nhiên, tôi không khuyên chuyển nguyên báo cáo hiện tại sang format paper rồi submit. Điểm yếu lớn nhất hiện nay không phải kết quả CER, mà là novelty và experimental protocol chưa đủ thuyết phục.

Đánh giá nhanh bản hiện tại
Thành phần	Hiện tại	Mức cho SOICT
Bài toán	Vietnamese air-writing	Tốt
Dataset	Vni_air_writing, 22,760 samples	Khá tốt
Mô hình	Conformer + CTC	Trung bình
Feature	RFF + trajectory dynamics	Có tiềm năng
Relative positional encoding	Có	Khá
Augmentation	Rotation, scaling, noise, time warping	Khá
Ablation	4 configurations	Có nhưng còn yếu
Baseline	Chưa đầy đủ	Yếu
Evaluation protocol	Có vấn đề	Cần sửa ngay
Novelty	Chưa được định nghĩa rõ	Yếu
Khả năng submit ngay	Chưa	
Khả năng sau khi phát triển	Khá cao	

Điểm đáng giữ nhất là nhóm đã không chỉ đưa raw \((x,y)\) vào mạng. Báo cáo đã khai thác velocity/direction/curvature và Random Fourier Features để biểu diễn trajectory, sau đó dùng Conformer + CTC.

Kết quả hiện tại cũng có tín hiệu khá tốt: CER giảm từ 17.32% → 15.79% khi absolute PE được thay bằng relative PE; cấu hình RFF + relative PE đạt 15.02%. Nhưng đáng chú ý là RFF + absolute PE lại xấu hơn baseline, 18.43%. Điều này thực ra có thể trở thành một kết quả nghiên cứu thú vị nếu được phân tích đúng, thay vì viết đơn giản rằng RFF luôn cải thiện.

Vấn đề lớn nhất cần sửa: experimental protocol

Trong báo cáo hiện tại, train lấy 1-gram, còn 2/3/n-gram được tách thành từ đơn để tạo validation/test. Sau đó nhóm augmentation cả validation và test, tăng mỗi từ từ 5 lên 25 mẫu rồi mới stratified split 20/80.

Đây là điểm reviewer rất dễ bắt.

Không nên augmentation validation/test. Augmentation chỉ được áp dụng cho training set. Nếu các biến thể của cùng một trajectory gốc đi vào cả validation và test thì kết quả có nguy cơ bị optimistic do sample correlation/leakage.

Tôi sẽ yêu cầu SV làm lại toàn bộ experiment trước khi viết paper.

Tôi đề xuất biến bài thành nghiên cứu về trajectory representation

Thay vì claim:

“We apply Conformer to Vietnamese air-writing.”

novelty như vậy khá yếu, vì Conformer và CTC đều là kỹ thuật đã biết.

Tôi đề xuất định vị thành:

A trajectory-aware representation framework for Vietnamese air-writing recognition

Trong đó contribution nằm ở biểu diễn trajectory, không nằm đơn thuần ở Conformer.

Pipeline nên thành:

$$ (x_t,y_t) \rightarrow \text{trajectory normalization} \rightarrow \begin{cases} \text{spatial representation}\\ \text{dynamic representation}\\ \text{spectral representation} \end{cases} \rightarrow \text{Conformer} \rightarrow CTC. $$
Spatial branch

Giữ:

$$ x_t,\quad y_t $$

nhưng tôi không khuyến nghị thay hoàn toàn \(x,y\) bằng RFF như bản hiện tại.

Dynamic branch

Từ trajectory tính:

$$ \Delta x_t,\Delta y_t $$ $$ v_t=\sqrt{\Delta x_t^2+\Delta y_t^2} $$ $$ \theta_t= \operatorname{atan2}(\Delta y_t,\Delta x_t) $$

và curvature:

$$ \kappa_t = \frac{x'_ty''_t-y'_tx''_t} {(x'^2_t+y'^2_t)^{3/2}+\epsilon}. $$

Đây là các đặc trưng có ý nghĩa vật lý trực tiếp đối với air-writing.

Spectral branch

RFF:

$$ \phi(p_t)= [\sin(2\pi Bp_t),\cos(2\pi Bp_t)] $$

với

$$ p_t=[x_t,y_t]. $$

Nhưng thay vì fixed/random RFF duy nhất như hiện nay, tôi muốn SV thử:

Raw XY + Dynamics + Multi-scale Fourier Encoding.

Ví dụ:

$$ \Phi(p)= [\sin(2^0\pi p),\cos(2^0\pi p),..., \sin(2^{K-1}\pi p),\cos(2^{K-1}\pi p)]. $$

Điều này tạo ra một contribution thuật toán rõ hơn.

Một cải tiến nữa rất đáng làm: gated feature fusion

Thay vì concatenate tất cả feature:

$$ z_t=[S_t;D_t;F_t] $$

có thể xây dựng ba embedding:

$$ h_s=f_s(S_t) $$ $$ h_d=f_d(D_t) $$ $$ h_f=f_f(F_t) $$

và học trọng số:

$$ [\alpha_s,\alpha_d,\alpha_f] = Softmax(W[h_s;h_d;h_f]). $$

Sau đó:

$$ h_t= \alpha_s h_s+ \alpha_d h_d+ \alpha_f h_f. $$

Rồi mới:

$$ h_t \rightarrow Conformer \rightarrow CTC. $$

Như vậy paper không còn là:

Conformer + CTC

mà trở thành:

Trajectory-aware multi-representation fusion + Conformer-CTC.

Đây là framing tốt hơn đáng kể.

Experiment tôi cho rằng bắt buộc phải có

Tôi sẽ thiết kế paper quanh 3 research questions.

RQ1 — Architecture: Conformer có thực sự phù hợp hơn các sequence models thông thường cho Vietnamese air-writing?

So sánh ít nhất:

Model	CER ↓	WER ↓	Params
BiLSTM-CTC			
GRU-CTC			
Transformer-CTC			
TCN-CTC			
Conformer-CTC			
Proposed			

Báo cáo hiện nói Conformer tốt hơn LSTM/Transformer nhưng bảng kết quả hiện có chỉ trình bày bốn cấu hình Fourier/positional encoding; vì vậy claim baseline hiện chưa được bảng kết quả hỗ trợ đầy đủ.

RQ2 — Representation: trajectory representation nào hiệu quả?

Đây nên là ablation quan trọng nhất:

Representation	CER
XY	
XY + velocity	
XY + velocity + direction	
XY + dynamics + curvature	
RFF only	
XY + RFF	
XY + dynamics + RFF	
Proposed fusion	

Bảng hiện tại mới chỉ cho thấy tương tác khá thú vị giữa RFF và positional encoding.

RQ3 — Robustness: mô hình có bền vững với biến thiên air-writing không?

Test riêng:

$$ rotation=\pm5^\circ,\pm10^\circ,\pm15^\circ $$

noise:

$$ \sigma=0.01,0.02,0.05 $$

scale:

$$ 0.8,0.9,1.1,1.2 $$

và temporal warping.

Sau đó plot:

CER vs perturbation strength.

Phần này rất hợp với bản chất bài toán vì chính báo cáo xác định rotation, hand jitter và writing-speed variation là các vấn đề thực tế.

Một ablation rất nên làm

Pipeline hiện có:

Savitzky–Golay → spline → normalization → resample 128.

Hãy thử:

Preprocessing	CER
Raw	
+ normalization	
+ Savitzky–Golay	
+ spline	
+ resampling	
Full pipeline	

Như vậy paper chứng minh được từng thành phần thực sự có tác dụng, thay vì chỉ mô tả pipeline.

Tôi còn muốn bổ sung một experiment về độ dài chuỗi

Hiện tất cả trajectory bị resample về 128 points.

Test:

$$ T=\{64,96,128,192,256\}. $$

Báo cáo:

CER – FLOPs – inference time.

Nếu 128 thực sự tối ưu thì ta có justification. Nếu 96 gần bằng 128 nhưng nhanh hơn nhiều thì lại tạo được câu chuyện efficient air-writing recognition.

Error analysis tiếng Việt sẽ làm bài mạnh hơn

Bản hiện tại đã phát hiện một hiện tượng rất hữu ích: model hay bỏ hoặc nhầm ký tự “i”, và một số nét chuyển tiếp làm mất thông tin.

Đừng bỏ phần này.

Tôi sẽ biến nó thành một subsection:

Error Analysis on Vietnamese Air-Writing

Phân nhóm:

dấu thanh;
nguyên âm có dấu;
short strokes như i;
repeated/overlapping trajectories;
epenthetic strokes;
long words;
fast writing.

Có thể báo cáo CER theo nhóm ký tự. Đây là điểm làm bài có “Vietnamese-specific insight”, thay vì chỉ là áp dụng model lên một dataset Việt Nam.

Contribution nên viết lại

Tôi đề xuất paper claim ba contribution:

1. We introduce a trajectory-aware representation for Vietnamese air-writing that jointly models spatial coordinates, motion dynamics, and multi-scale spectral features.

2. We develop a Conformer-CTC recognition framework with relative positional modeling and adaptive fusion of complementary trajectory representations.

3. We conduct systematic evaluations across sequence architectures, feature representations, preprocessing strategies, and trajectory perturbations, providing an analysis of robustness and Vietnamese-specific recognition errors.

Ba contribution này đủ dáng một conference paper hơn nhiều so với “dùng Conformer + CTC”.

Tên bài tôi ưu tiên

Tôi không dùng tên chung chung kiểu Vietnamese Air-Writing Recognition Using Conformer.

Tôi thích:

Trajectory-Aware Conformer for Vietnamese Air-Writing Recognition

Nếu làm gated/multi-representation fusion:

Multi-Representation Trajectory Learning for Vietnamese Air-Writing Recognition

Tên thứ hai theo tôi mạnh hơn, vì không đóng novelty vào một backbone có sẵn.

Cấu trúc paper SOICT

Với giới hạn của SOICT, tôi sẽ viết khoảng 10–12 trang:

1. Introduction — 1–1.5 trang
Problem → challenge → research gap → proposed idea → contributions.

2. Related Work — 1 trang
Air-writing recognition → trajectory representation → sequence recognition → Conformer/CTC.

3. Proposed Method — 2.5–3 trang
Preprocessing → spatial/dynamic/spectral representation → fusion → Conformer → CTC.

4. Experimental Setup — 1–1.5 trang
Dataset → clean split → baselines → metrics → implementation.

5. Results and Discussion — 3–4 trang
Main comparison → feature ablation → architecture ablation → robustness → error analysis.

6. Conclusion — khoảng 0.5 trang.

Mức khả năng accept

Nếu submit gần như nội dung hiện tại, tôi chỉ đánh giá khoảng 4–5/10 về độ thuyết phục: topic phù hợp nhưng novelty chưa rõ, baseline thiếu và protocol validation/test có điểm reviewer rất dễ phản biện.

Nếu sửa split, thêm baseline + ablation + robustness nhưng giữ Conformer/RFF hiện tại, tôi đánh giá khoảng 6.5–7/10.

Nếu triển khai hướng multi-representation trajectory learning + gated fusion, làm clean evaluation, baseline đầy đủ và error analysis tốt, tôi xem đây là một bài khá phù hợp SOICT, khoảng 7.5–8/10 về chất lượng mục tiêu. Đây không phải xác suất accept thực tế vì còn phụ thuộc reviewer và mặt bằng submission, nhưng là mức mà tôi sẽ thấy đủ tự tin để gửi.

Điểm thuận lợi là SOICT 2026 chấp nhận cả theoretical, empirical và experimental CS research, và track Multimedia Processing explicitly bao gồm pattern recognition, AR/VR và human-centered multimedia computing.

Quan trọng nhất: hiện là 06/09, abstract deadline 09/09 và full paper 16/09/2026. Vì vậy tôi sẽ không mở rộng sang video end-to-end/MediaPipe lúc này dù báo cáo đề xuất hướng đó. Nó làm scope phình quá lớn. Với khoảng 10 ngày còn lại, hướng khả thi nhất là giữ dataset + Conformer-CTC, sửa protocol, phát triển trajectory representation/fusion và chạy một ma trận thực nghiệm có chủ đích. Đây vừa đủ novelty vừa có khả năng hoàn thành kịp deadline.

SOICT 2026 – Paper Submission and Scope

Sources
bận đề xuất nội dung chỉnh sửa cụ thể và nội dung thực nghiệm cụ thể để thực hiện

Được. Với bản hiện tại, tôi đề xuất không viết lại từ đầu, mà giữ lõi “trajectory + Conformer + CTC”, sau đó chỉnh theo hướng Multi-Representation Trajectory Learning for Vietnamese Air-Writing Recognition. Mục tiêu là tạo ra một bài có novelty rõ, protocol sạch, baseline đủ và ablation thuyết phục.

Hiện báo cáo đã có nền tảng khá tốt: dữ liệu là chuỗi tọa độ 2D theo thời gian; có Savitzky–Golay, spline, resampling về 128 điểm; có các đặc trưng vận tốc/hướng/độ cong; có RFF; backbone là Conformer + CTC; và đã có một ablation nhỏ giữa Fourier feature và positional encoding.

Tôi sẽ yêu cầu sinh viên phát triển theo nội dung rất cụ thể dưới đây.

1. Định hướng lại contribution của bài

Không nên claim:

“Đề xuất Conformer + CTC cho air-writing.”

Vì Conformer và CTC đều là kỹ thuật có sẵn.

Nên chuyển thành:

Đề xuất một biểu diễn trajectory đa thành phần kết hợp thông tin hình học, động học và phổ cho nhận dạng chữ viết trên không tiếng Việt, sử dụng Conformer-CTC làm bộ mã hóa chuỗi.

Tức contribution chính nằm ở:

$$ \text{Spatial} + \text{Dynamic} + \text{Spectral} \rightarrow \text{Fusion} \rightarrow \text{Conformer} \rightarrow CTC $$

Tên bài tôi ưu tiên:

Multi-Representation Trajectory Learning for Vietnamese Air-Writing Recognition

hoặc nếu muốn giữ “Conformer” trong title:

Trajectory-Aware Conformer for Vietnamese Air-Writing Recognition

Tôi nghiêng về tên đầu tiên.

2. Chỉnh phần dữ liệu và protocol trước tiên

Đây là phần phải sửa trước khi chạy bất kỳ mô hình mới nào.

Bản hiện tại lấy 1-gram làm train, sau đó tách các cụm 2/3/n-gram thành từ đơn để tạo validation/test. Quan trọng hơn, báo cáo hiện áp dụng augmentation cả validation và test rồi mới chia.

Phần này nên sửa thành:

Training set

Chỉ training mới được augmentation.

Ví dụ:

$$ D_\text{train}^{aug} = D_\text{train} + R(D) + S(D) + N(D) + T(D) $$

với:

\(R\): rotation
\(S\): scaling
\(N\): Gaussian noise
\(T\): temporal warping
Validation set

Không augmentation.

Test set

Không augmentation.

Nếu dataset có thông tin người viết, cần ưu tiên:

$$ Writer_{train} \cap Writer_{test} = \varnothing $$

Nếu dataset không có writer ID thì phải nói rõ đây là limitation.

Quan trọng

Các mẫu được tạo ra từ cùng một trajectory gốc tuyệt đối không được phân phối sang các tập khác nhau.

Ví dụ mẫu:

sample_001.csv
sample_001_rotate.csv
sample_001_noise.csv
sample_001_warp.csv

phải nằm hoàn toàn trong training nếu sample_001 thuộc training.

3. Chỉnh phần preprocessing

Pipeline hiện tại:

$$ Raw \rightarrow Savitzky-Golay \rightarrow Spline \rightarrow Normalization \rightarrow Resample(128) $$

là hợp lý và có thể giữ lại.

Nhưng paper phải mô tả toán học rõ hơn.

Với trajectory:

$$ P=\{p_t\}_{t=1}^{T}, \qquad p_t=(x_t,y_t) $$

sau translation:

$$ \tilde p_t=p_t-p_1 $$

hoặc centroid normalization:

$$ \tilde p_t=p_t-\frac1T\sum_{i=1}^{T}p_i. $$

Sau đó scale:

$$ \hat p_t= \frac{\tilde p_t} {\max_i\|\tilde p_i\|_2+\epsilon}. $$

Tôi khuyến nghị dùng centroid + global scale, thay vì “đưa điểm dưới cùng bên phải thành gốc” như cách viết trong báo cáo hiện tại. Cách centroid dễ giải thích và ít phụ thuộc orientation hơn.

4. Xây dựng 3 nhóm đặc trưng

Đây sẽ là phần thuật toán chính của paper.

A. Spatial representation

Giữ tọa độ:

$$ S_t=[x_t,y_t]. $$

Không nên bỏ hoàn toàn \(x,y\) như phiên bản hiện tại đã thử với RFF.

B. Dynamic representation

Tính displacement:

$$ \Delta x_t=x_t-x_{t-1}, $$ $$ \Delta y_t=y_t-y_{t-1}. $$

Speed:

$$ v_t= \sqrt{ \Delta x_t^2+ \Delta y_t^2 }. $$

Direction:

$$ d_t= \left[ \frac{\Delta x_t}{v_t+\epsilon}, \frac{\Delta y_t}{v_t+\epsilon} \right]. $$

Acceleration:

$$ a_t=v_t-v_{t-1}. $$

Curvature:

$$ \kappa_t= \frac{ x'_t y''_t-y'_t x''_t }{ (x'^2_t+y'^2_t)^{3/2}+\epsilon }. $$

Dynamic feature:

$$ D_t= [ \Delta x_t, \Delta y_t, v_t, d_t^x, d_t^y, a_t, \kappa_t ]. $$

Báo cáo hiện tại đã khai thác đạo hàm, hướng và curvature; đây là cơ sở tốt để phát triển phần này.

5. Sửa Random Fourier Feature thành multi-scale Fourier encoding

RFF hiện tại có tín hiệu nhưng kết quả chưa ổn định. Cụ thể:

No RFF + absolute PE: 17.32%
No RFF + relative PE: 15.79%
RFF + absolute PE: 18.43%
RFF + relative PE: 15.02%

Điều này cho thấy RFF không tự động cải thiện kết quả.

Tôi đề xuất chuyển từ một mapping RFF đơn sang multi-scale Fourier encoding.

Với:

$$ p_t=[x_t,y_t] $$

tạo:

$$ F_t= [ \sin(2^0\pi p_t), \cos(2^0\pi p_t), ... $$ $$ \sin(2^{K-1}\pi p_t), \cos(2^{K-1}\pi p_t) ]. $$

Thử:

$$ K\in\{2,4,6,8\}. $$

Đây là experiment rất dễ làm nhưng có giá trị.

6. Đề xuất module mới: Multi-Representation Fusion

Đây là phần tôi khuyên thêm để tạo novelty.

Ba representation:

$$ S_t,\quad D_t,\quad F_t $$

được embedding riêng:

$$ h_t^S=W_S S_t+b_S $$ $$ h_t^D=W_D D_t+b_D $$ $$ h_t^F=W_F F_t+b_F. $$

Cách đơn giản nhất:

$$ h_t= [h_t^S;h_t^D;h_t^F] $$

rồi:

$$ z_t=W_fh_t+b_f. $$

Đây sẽ là baseline fusion.

Sau đó đưa ra phương pháp đề xuất adaptive gated fusion:

$$ g_t= Softmax ( W_g[h_t^S;h_t^D;h_t^F] ). $$

Với:

$$ g_t= [ \alpha_t^S, \alpha_t^D, \alpha_t^F ]. $$

Feature cuối:

$$ h_t= \alpha_t^S h_t^S + \alpha_t^D h_t^D + \alpha_t^F h_t^F. $$

Sau đó:

$$ H= \{h_t\}_{t=1}^{T} \rightarrow Conformer \rightarrow Linear \rightarrow CTC. $$

Như vậy module thực sự được “đề xuất” là:

Adaptive Multi-Representation Trajectory Fusion.

Có thể đặt tên ngắn:

AMTF

và toàn mô hình:

AMTF-Conformer.

7. Giữ Relative Positional Encoding

Relative PE đang cho kết quả tốt rõ ràng hơn absolute PE:

$$ 17.32\%\rightarrow15.79\% $$

khi không có Fourier, và:

$$ 18.43\%\rightarrow15.02\% $$

khi có Fourier.

Do đó relative positional encoding nên trở thành default configuration, còn absolute PE chỉ dùng cho ablation.

Không nên claim relative PE là contribution mới.

8. Kiến trúc cuối cùng

Framework nên là:

Raw trajectory
      │
      ▼
Savitzky-Golay
      │
      ▼
Spline + normalization
      │
      ▼
Uniform resampling
      │
      ├───────────────┬─────────────────┐
      ▼               ▼                 ▼
Spatial            Dynamic            Fourier
x,y          dx,dy,v,dir,a,k       Multi-scale FE
      │               │                 │
      ▼               ▼                 ▼
 Spatial FC       Dynamic FC        Fourier FC
      │               │                 │
      └───────────────┴─────────────────┘
                      │
                      ▼
             Adaptive Fusion
                      │
                      ▼
             Relative Pos. Enc.
                      │
                      ▼
             Conformer Encoder
                      │
                      ▼
                 Linear
                      │
                      ▼
                   CTC
                      │
                      ▼
             Character sequence

Đây nên là Figure 1 của paper.

9. Experimental protocol cụ thể

Tôi đề xuất sinh viên chạy theo thứ tự sau để tránh tốn GPU vô ích.

Experiment 1 — Baseline architectures

Mục đích:

Conformer có thực sự là backbone phù hợp?

Giữ input giống nhau:

$$ [x,y,\Delta x,\Delta y] $$

và cùng preprocessing.

Chạy:

ID	Model
B1	BiLSTM + CTC
B2	GRU + CTC
B3	TCN + CTC
B4	Transformer + CTC
B5	Conformer + CTC

Cố gắng để embedding dimension gần tương đương.

Báo cáo:

Model	CER ↓	WER ↓	Params	FLOPs	Time/sample

Nếu dataset chủ yếu là word-level, ngoài CER nên có:

$$ Word\ Accuracy = \frac{N_{correct}}{N}. $$

Tôi khuyên dùng cả:

CER + Word Accuracy.

10. Experiment 2 — Feature ablation

Đây là experiment quan trọng nhất.

Giữ Conformer cố định.

Chạy:

ID	Feature
F1	XY
F2	XY + ΔXY
F3	XY + ΔXY + speed
F4	XY + ΔXY + speed + direction
F5	XY + dynamics + curvature
F6	Fourier only
F7	XY + Fourier
F8	Dynamics + Fourier
F9	XY + Dynamics + Fourier

Expected paper table:

Features	CER ↓	Word Acc. ↑
XY		
XY + ΔXY		
XY + Dynamics		
Fourier		
XY + Fourier		
Dynamics + Fourier		
Spatial + Dynamic + Spectral	...	...

Nếu F9 tốt nhất, ta đã chứng minh được multi-representation.

11. Experiment 3 — Fusion strategy

Tiếp theo so sánh:

Simple concatenation
$$ h=[h_S;h_D;h_F] $$
Element-wise sum
$$ h=h_S+h_D+h_F $$
Static weighted sum
$$ h= \alpha h_S+ \beta h_D+ \gamma h_F $$
Proposed adaptive gated fusion
$$ h_t= \alpha_t^S h_t^S+ \alpha_t^D h_t^D+ \alpha_t^F h_t^F. $$

Bảng:

Fusion	CER ↓	Acc. ↑
Concatenation		
Sum		
Static weighting		
Adaptive gated fusion	...	...

Đây sẽ là bảng chứng minh contribution chính.

12. Experiment 4 — Positional encoding

Giữ mọi thứ như proposed model.

So sánh:

PE
No positional encoding
Absolute PE
Relative PE

Bảng hiện tại đã có hai trường hợp cuối, chỉ cần bổ sung no-PE.

13. Experiment 5 — Fourier scale

Nếu dùng:

$$ K=\{2,4,6,8\} $$

thì chạy:

K	Fourier dim	CER
0	0	
2		
4		
6		
8		

Experiment này giúp reviewer thấy Fourier dimension không được chọn tùy tiện.

14. Experiment 6 — Preprocessing ablation

Từ pipeline hiện tại, chạy:

Setting	SG Filter	Spline	Normalize	CER
P1	✗	✗	✗	
P2	✓	✗	✓	
P3	✗	✓	✓	
P4	✓	✓	✓	

Không cần chạy quá nhiều combination.

Mục đích là chứng minh preprocessing không chỉ “thêm cho có”.

15. Experiment 7 — Resampling length

Hiện báo cáo chọn 128 điểm nhưng chưa chứng minh đây là lựa chọn tốt nhất.

Chạy:

$$ T= 64,\;96,\;128,\;192,\;256. $$

Bảng:

Length	CER	Params	FLOPs	Time
64				
96				
128				
192				
256				

Params sẽ gần như không đổi nhưng FLOPs/time sẽ tăng.

Nếu 128 là optimum thì giữ 128.

16. Experiment 8 — Data augmentation ablation

Hiện có 4 augmentation:

Rotation
Stretching
Gaussian noise
Time warping

Chạy:

Augmentation	CER
None	
Rotation	
Scaling	
Gaussian noise	
Time warping	
All	

Nếu có thời gian thêm:

All − Rotation
All − Scaling
All − Noise
All − Time warping

đây là leave-one-out ablation.

17. Experiment 9 — Robustness test

Phần này có giá trị rất cao đối với air-writing.

Quan trọng: đây là perturb test trên test set, không phải augmentation để tăng test samples rồi tính chung.

Rotation robustness
$$ \theta= 0^\circ, 5^\circ, 10^\circ, 15^\circ, 20^\circ. $$
Noise robustness

Ví dụ normalized coordinates:

$$ \sigma= 0,\; 0.005,\; 0.01,\; 0.02,\; 0.05. $$
Scale variation
$$ s= 0.8,\; 0.9,\; 1.0,\; 1.1,\; 1.2. $$
Temporal distortion

Mức:

none
weak
medium
strong

Plot:

$$ CER \text{ vs perturbation level}. $$

So sánh 3 mô hình:

Transformer
Conformer
Proposed AMTF-Conformer

Nếu proposed model degrade chậm hơn thì có một kết quả rất đẹp.

18. Experiment 10 — Error analysis tiếng Việt

Bản hiện tại đã quan sát thấy ký tự “i” dễ bị bỏ hoặc bị nhầm, và nét chuyển tiếp gây lỗi.

Phần này nên phát triển thành kết quả định lượng.

Tính error theo từng character:

$$ CER(c) $$

cho:

a, ă, â
e, ê
o, ô, ơ
u, ư
i
d, đ

Nếu label giữ dấu tiếng Việt, phân tích thêm:

á à ả ã ạ
ă ắ ằ ẳ ẵ ặ
â ấ ầ ẩ ẫ ậ
...

Có thể tạo confusion matrix hoặc Top-10 confused characters.

Ví dụ bảng:

GT	Most frequent prediction	Error rate
i	deletion	
ă	a	
â	a	
đ	d	

Đây là một phần rất đáng giá vì khiến bài có đặc trưng Vietnamese air-writing, chứ không chỉ “dataset Việt Nam”.

19. Experiment 11 — Word length analysis

Phân test thành:

2–3 characters
4–5 characters
6–7 characters
≥8 characters

Tính:

$$ CER_L. $$

Ví dụ:

Length	No. samples	CER
2–3		
4–5		
6–7		
≥8		

Nếu CER tăng theo word length thì phần discussion rất tự nhiên.

20. Nên chạy nhiều seed

Không nên báo đúng một số:

$$ CER=15.02\%. $$

Nên chạy ít nhất:

$$ N=3 $$

random seeds.

Báo:

$$ mean\pm std. $$

Ví dụ:

$$ 14.12\pm0.18\%. $$

Nếu GPU đủ thì 5 runs tốt hơn, nhưng với thời gian SOICT hiện tại, 3 runs là hợp lý.

21. Chỉ số đánh giá nên bổ sung

Hiện báo cáo chỉ dùng CER.

Tôi đề xuất:

$$ CER= \frac{S+D+I}{N}. $$

Thêm:

$$ WER= \frac{S_w+D_w+I_w}{N_w}. $$

Nếu đầu ra từng word riêng thì dùng:

$$ Word\ Accuracy. $$

Và báo riêng:

$$ Substitution\ Rate, $$ $$ Deletion\ Rate, $$ $$ Insertion\ Rate. $$

Air-writing đặc biệt dễ gặp deletion do các nét ngắn như “i”, nên decomposition này sẽ rất hữu ích.

22. Main result table cuối cùng nên như thế nào

Table quan trọng nhất của paper nên là:

Method	Features	CER ↓	Word Acc. ↑	Params
BiLSTM-CTC	XY			
GRU-CTC	XY			
TCN-CTC	XY			
Transformer-CTC	XY			
Conformer-CTC	XY			
Conformer-CTC	Spatial+Dynamic			
Conformer-CTC	Spatial+Dynamic+Fourier			
AMTF-Conformer	Adaptive multi-representation	...	...	

Reviewer chỉ cần nhìn bảng này đã hiểu contribution.

23. Các figure nên có

Paper không cần giữ 21 hình như báo cáo sinh viên.

Tôi chỉ giữ khoảng 5 hình:

Figure 1 — Proposed framework

Raw trajectory → representations → fusion → Conformer → CTC.

Figure 2 — Trajectory representations

Một trajectory và các feature:

$$ x,y,\quad v,\quad direction,\quad curvature. $$

Figure 3 — Training curves

Loss/CER train-validation.

Figure 4 — Robustness

CER vs rotation/noise.

Figure 5 — Error cases

Correct + deletion + confusion + epenthetic stroke.

Những hình mô tả từng module Conformer, attention, convolution như Hình 8–11 trong báo cáo hiện tại có thể bỏ, vì đó là kiến trúc đã có trong literature.

24. Nội dung Introduction nên đổi

Hiện introduction giải thích khá dài về HCI, VR/AR và air-writing. Phần đó có thể rút xuống khoảng 2 paragraph.

Sau đó đưa research gap rõ:

Existing air-writing methods mainly rely on raw coordinates or handcrafted motion descriptors. Raw coordinates preserve trajectory geometry but are sensitive to spatial variations, whereas dynamic descriptors emphasize motion patterns but may lose absolute structural information. Spectral encodings can capture fine trajectory variations, yet their complementary role with spatial and dynamic features has not been sufficiently investigated for Vietnamese air-writing recognition.

Sau đó:

To address this issue, we propose a multi-representation trajectory learning framework...

Đây mới là câu chuyện nghiên cứu.

25. Related Work nên chia 3 nhóm

Không nên viết dài.

Air-writing recognition

Các phương pháp camera/sensor/trajectory.

Sequence modeling

LSTM, Transformer, Conformer, CTC.

Trajectory representation

coordinate, derivative, dynamic descriptors, Fourier features.

Kết thúc bằng gap:

Existing methods have not systematically investigated the joint contribution of spatial, dynamic and spectral trajectory representations in Vietnamese air-writing.

26. Phần Method nên tổ chức lại

Tôi đề xuất các subsection:

III. Proposed Method

A. Trajectory Preprocessing

B. Spatial Representation

C. Dynamic Representation

D. Multi-scale Fourier Representation

E. Adaptive Multi-Representation Fusion

F. Conformer Encoder and CTC Decoding

Trong đó E là contribution quan trọng nhất.

27. Experimental section

Cấu trúc:

IV. Experiments

A. Dataset and Evaluation Protocol

B. Implementation Details

C. Comparison with Sequence Baselines

D. Ablation Study

E. Robustness Analysis

F. Error Analysis

Không cần tách quá nhiều subsection nhỏ.

28. Những experiment nào là bắt buộc và experiment nào có thể bỏ

Nếu thời gian ít, tôi ưu tiên theo thứ tự:

Priority	Experiment
Must	Clean train/val/test
Must	Baseline LSTM/Transformer/Conformer
Must	Spatial/Dynamic/Fourier ablation
Must	Fusion ablation
Must	Relative vs absolute PE
High	Augmentation ablation
High	Robustness
High	Error analysis
Medium	Sequence length
Medium	Fourier scale
Low	Extensive hyperparameter search

Nếu chỉ đủ thời gian chạy 6 thí nghiệm, tôi chọn:

Baseline architectures.
Feature ablation.
Fusion ablation.
Positional encoding.
Robustness.
Error analysis.
29. Kịch bản chạy thực nghiệm thực tế cho sinh viên

Tôi sẽ yêu cầu các em làm theo trình tự này:

Giai đoạn 1 — Freeze data

Tạo duy nhất:

train.txt
val.txt
test.txt

và không thay đổi nữa.

Lưu seed.

Giai đoạn 2 — Reproduce baseline hiện tại

Chạy lại:

Conformer
XY/basic dynamics
Relative PE
CTC

để có baseline sạch.

Không dùng augmented test.

Con số CER mới có thể sẽ xấu hơn 15.02%. Điều đó không sao; quan trọng là đúng protocol.

Giai đoạn 3 — Architecture baselines

Run:

BiLSTM
Transformer
Conformer

Nếu ít thời gian thì ba model này là đủ.

Giai đoạn 4 — Feature experiments

Run:

XY
XY + Dynamics
XY + Fourier
XY + Dynamics + Fourier

Nếu kết quả tốt thì tiếp tục.

Giai đoạn 5 — Proposed fusion

Run:

Concatenation
Adaptive gated fusion

Nếu gated tốt hơn rõ thì contribution đã hình thành.

Giai đoạn 6 — Robustness

Chỉ chạy:

Baseline Conformer
Proposed

trên:

clean,
rotation,
noise,
time warp.
Giai đoạn 7 — Error analysis

Không cần train thêm.

Chỉ lấy prediction của proposed model và phân tích.

30. Một ma trận thực nghiệm tối giản

Nếu muốn tiết kiệm thời gian tối đa thì toàn bài chỉ cần khoảng 12–15 configuration:

1  BiLSTM
2  Transformer
3  Conformer

4  Conformer + XY
5  Conformer + XY+Dynamics
6  Conformer + XY+Fourier
7  Conformer + XY+Dynamics+Fourier

8  Proposed concat
9  Proposed gated

10 Absolute PE
11 Relative PE

12 No augmentation
13 Full augmentation

Sau đó robustness dùng model đã train, không cần train thêm.

Nếu mỗi model chạy 3 seed thì khoảng 39 runs. Đây vẫn là khối lượng hợp lý.

31. Điều tôi không khuyến nghị làm lúc này

Báo cáo hiện đề xuất tương lai phát triển end-to-end từ video và MediaPipe.

Tôi không làm cho bài SOICT này.

Không nên đồng thời thêm:

YOLO/MediaPipe;
video recognition;
language model;
self-supervised learning;
3D trajectory;
multimodal sensor.

Như vậy bài sẽ bị loãng.

Bài này nên có một câu chuyện duy nhất:

How should a Vietnamese air-writing trajectory be represented for sequence recognition?

Và answer:

Joint spatial, dynamic and spectral representation with adaptive fusion.

Đó là câu chuyện rõ nhất và có thể triển khai nhanh nhất từ code hiện tại.

Mục tiêu kết quả nên đặt ra

Hiện best CER báo cáo là 15.02%, nhưng protocol cần làm lại. Vì vậy không nên lấy 15.02% làm mốc bắt buộc.

Sau khi clean split, giả sử baseline Conformer ra:

$$ CER=17.0\%. $$

Nếu proposed đạt khoảng:

$$ 14.5\%-15.0\% $$

tức relative improvement khoảng:

$$ \frac{17-14.5}{17}\approx14.7\% $$

thì rất đẹp.

Ngay cả khi:

$$ 17.0\%\rightarrow15.8\% $$

nhưng improvement ổn định 3 seeds, cộng với robustness tốt hơn và ablation đầy đủ thì vẫn hoàn toàn viết được bài tốt.

Tôi ưu tiên protocol + consistency + ablation hơn cố ép CER xuống một con số thật thấp.

Nếu thực hiện theo hướng này, phần việc của sinh viên thực chất không quá lớn: không thay backbone chính, chỉ sửa data split, bổ sung các feature branch, thêm gated fusion và chạy một ma trận experiment có hệ thống. Đây là phương án tôi cho là cân bằng tốt nhất giữa novelty – thời gian thực hiện – khả năng accept SOICT.