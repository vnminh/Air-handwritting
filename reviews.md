Có. Tôi đã kiểm tra trực tiếp code trong Air-writing-soict-main.zip. Với code hiện tại, tôi đề xuất không chạy lại hàng loạt backbone trước. Nên tập trung vào các thực nghiệm trả lời trực tiếp novelty của bài: representation → adaptive fusion → robustness → gate behavior → lỗi đặc thù tiếng Việt.

Một lưu ý rất quan trọng trước khi chạy: trong code hiện tại, trajectory được arc-length resample về 128 điểm trước, sau đó mới tính speed, acceleration trong dynamic_features(). Vì vậy các đại lượng này không còn phản ánh hoàn toàn “writing speed” theo thời gian gốc; chúng gần hơn với local geometric motion descriptors. Trong bài hiện tại, phần methodology đang gọi chúng là speed/dynamics. 
SOICT Tôi khuyến nghị trước mắt giữ pipeline để bảo toàn kết quả 1.77%, nhưng khi viết bài nên dùng từ thận trọng hơn, hoặc bổ sung thí nghiệm temporal-vs-arc-length ở cuối.

1. Bộ thực nghiệm tôi đề xuất

Tôi chia thành 5 experiment chính. Nếu hoàn thành đủ 5 nhóm này, phần Results sẽ thuyết phục hơn đáng kể.

Exp.	Mục tiêu	Có cần train lại?	Kết quả đưa vào paper
E1	Representation ablation	Có	Table II
E2	Fusion ablation	Có	Table III
E3	Multi-seed confirmation	Có	Mean ± std
E4	Robustness	Không với checkpoint đã train	Fig. CER vs perturbation
E5	Gate + Vietnamese error analysis	Không	Fig. + Table IV

Ngoài ra tôi đề xuất E6 – lexical-disjoint recognition nếu thời gian GPU cho phép. Nó có thể làm bài mạnh lên đáng kể vì chứng minh mô hình nhận chuỗi ký tự chứ không chỉ ghi nhớ 660 labels.

2. E1 — Representation Ablation

Đây là thực nghiệm bắt buộc.

Câu hỏi cần trả lời:

Spatial, geometric-motion và Fourier information thực sự bổ sung cho nhau hay không?

Các cấu hình cần chạy:

Spatial only: xy
Dynamic only
Fourier only: fourier
Spatial + Dynamic: xy_dynamics
Spatial + Fourier: xy_fourier
Dynamic + Fourier: dynamics_fourier
Spatial + Dynamic + Fourier: all

Tất cả giữ cố định:

backbone="conformer"
fusion="gated"
positional="relative"
fourier_scales=2
length=128
augment=True
seed=42
Vấn đề nhỏ trong code hiện tại

Code đã hỗ trợ gần như toàn bộ, nhưng chưa hỗ trợ dynamic-only.

Trong models.py, hàm:

def branch_dimensions(...)

hiện tại có:

if representation in {"xy", "xy_delta", "xy_dynamics", "xy_fourier", "all"}:
    dimensions["spatial"] = 2

if representation == "xy_delta":
    dimensions["dynamic"] = 2
elif representation in {"xy_dynamics", "dynamics_fourier", "all"}:
    dimensions["dynamic"] = 7

hãy sửa thành:

def branch_dimensions(representation: str, fourier_scales: int) -> dict[str, int]:
    dimensions = {}

    if representation in {
        "xy",
        "xy_delta",
        "xy_dynamics",
        "xy_fourier",
        "all",
    }:
        dimensions["spatial"] = 2

    if representation == "xy_delta":
        dimensions["dynamic"] = 2

    elif representation in {
        "dynamics",
        "xy_dynamics",
        "dynamics_fourier",
        "all",
    }:
        dimensions["dynamic"] = 7

    if representation in {
        "fourier",
        "xy_fourier",
        "dynamics_fourier",
        "all",
    }:
        dimensions["fourier"] = 4 * fourier_scales

    if not dimensions:
        raise ValueError(f"Unknown representation: {representation}")

    return dimensions

Sau đó trong data.py, sửa:

elif representation in {"xy_dynamics", "dynamics_fourier", "all"}:

thành:

elif representation in {
    "dynamics",
    "xy_dynamics",
    "dynamics_fourier",
    "all",
}:

Như vậy ta mới có dynamic-only baseline hoàn chỉnh.

3. Tạo script chạy paper experiments

Hiện repository có run_experiment() nhưng không có CLI training hoàn chỉnh. Tôi khuyên tạo một file:

paper_experiments.py

ở root repository.

Nội dung:

import json
from pathlib import Path

from airwriting.experiment import ExperimentConfig, run_experiment


DATA_ROOT = Path("/kaggle/input/YOUR_DATASET")
MANIFEST_PATH = Path("manifest.json")
OUTPUT_ROOT = Path("paper_results")


with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)


BASE = dict(
    backbone="conformer",
    positional="relative",
    fourier_scales=2,
    length=128,
    savgol=True,
    spline=True,
    normalize=True,
    augment=True,

    d_model=128,
    layers=4,
    heads=4,
    d_ff=512,
    dropout=0.1,

    batch_size=512,
    epochs=40,
    patience=20,
    learning_rate=1e-3,
    weight_decay=1e-2,

    seed=42,
)


representation_configs = [
    ("rep_spatial", "xy", "gated"),
    ("rep_dynamic", "dynamics", "gated"),
    ("rep_fourier", "fourier", "gated"),

    ("rep_spatial_dynamic", "xy_dynamics", "gated"),
    ("rep_spatial_fourier", "xy_fourier", "gated"),
    ("rep_dynamic_fourier", "dynamics_fourier", "gated"),

    ("rep_all", "all", "gated"),
]


for name, representation, fusion in representation_configs:

    config = ExperimentConfig(
        name=name,
        representation=representation,
        fusion=fusion,
        **BASE,
    )

    result = run_experiment(
        config=config,
        data_root=DATA_ROOT,
        manifest=manifest,
        output_root=OUTPUT_ROOT,
    )

    print(
        name,
        result["test"]["cer"],
        result["test"]["wer"],
        result["test"]["exact_accuracy"],
    )

Sau đó:

python paper_experiments.py

Mỗi experiment sẽ tự tạo:

paper_results/
    rep_spatial/
        seed_42/
            best.pt
            result.json
            predictions_test.jsonl

Rất thuận tiện để tổng hợp paper.

4. Table II cần lấy kết quả gì?

Từ từng:

result.json

lấy:

result["parameters"]

result["test"]["cer"]
result["test"]["wer"]
result["test"]["exact_accuracy"]

Bảng cuối nên là:

Representation	CER ↓	WER ↓	Exact ↑	Params
Spatial				
Dynamic				
Fourier				
Spatial + Dynamic				
Spatial + Fourier				
Dynamic + Fourier				
S + D + F	1.77	5.55	94.77	1.55 M

Đây là bảng chứng minh contribution đầu tiên.

5. E2 — Fusion Ablation

Đây thậm chí còn quan trọng hơn E1, vì novelty của paper chúng ta muốn đặt ở:

representation-adaptive fusion

Code BranchFusion hiện tại đã hỗ trợ đủ:

concat
sum
static
gated

nên không cần sửa model.

Chạy 4 cấu hình:

fusion_configs = [
    ("fusion_sum", "sum"),
    ("fusion_concat", "concat"),
    ("fusion_static", "static"),
    ("fusion_gated", "gated"),
]

for name, fusion in fusion_configs:

    config = ExperimentConfig(
        name=name,

        representation="all",
        fusion=fusion,

        **BASE,
    )

    run_experiment(
        config,
        DATA_ROOT,
        manifest,
        OUTPUT_ROOT,
    )

Điều kiện phải tuyệt đối giữ giống nhau:

Representation = all
Fourier K = 2
Backbone = Conformer
Position = relative
Augmentation = yes
Seed = 42

Sau đó tạo:

Table III. Effect of fusion strategy
Fusion	CER ↓	WER ↓	Exact ↑	Params
Mean/Sum				
Concatenation				
Static learned weighting				
Time-dependent gated				

Nếu gated tốt nhất một cách ổn định thì novelty được củng cố rất mạnh.

Nếu gated chỉ hơn concat rất ít, vẫn giữ được, nhưng claim chuyển thành:

adaptive fusion provides comparable or improved recognition while enabling trajectory-level representation analysis.

Không nên ép claim accuracy nếu dữ liệu không chứng minh.

6. E3 — Multi-seed confirmation

Hiện paper chủ yếu báo seed 42. Đây là một điểm reviewer có thể hỏi.

Bản hiện tại cho biết các experiment dùng fixed manifest và seed 42 trừ khi có multi-seed confirmation. 
SOICT

Tôi đề xuất không chạy multi-seed cho tất cả model vì tốn GPU.

Chỉ chạy 3 cấu hình:

Conformer spatial baseline
All + concat
All + gated proposed

với 3 hoặc 5 seeds.

Đối với SOICT, 3 seeds là đủ hợp lý:

SEEDS = [42, 3407, 2026]

Có thể dùng:

for seed in [42, 3407, 2026]:

    config = ExperimentConfig(
        name="proposed_multiseed",
        representation="all",
        fusion="gated",
        seed=seed,

        backbone="conformer",
        positional="relative",
        fourier_scales=2,

        length=128,
        augment=True,

        d_model=128,
        layers=4,
        heads=4,
        d_ff=512,

        batch_size=512,
        epochs=40,
        patience=20,
    )

    run_experiment(
        config,
        DATA_ROOT,
        manifest,
        OUTPUT_ROOT,
    )

Quan trọng:

Không tạo lại manifest theo từng seed.

Dataset split phải giữ cố định, chỉ seed initialization/training thay đổi.

Nếu không, anh đang đo cả variation do split.

7. Tổng hợp mean ± std tự động

Sau khi chạy xong:

import json
import numpy as np
from pathlib import Path

root = Path("paper_results/proposed_multiseed")

cers = []
wers = []
exacts = []

for result_file in root.glob("seed_*/result.json"):

    with open(result_file, encoding="utf-8") as f:
        r = json.load(f)

    cers.append(r["test"]["cer"] * 100)
    wers.append(r["test"]["wer"] * 100)
    exacts.append(r["test"]["exact_accuracy"] * 100)


def report(name, values):
    print(
        f"{name}: "
        f"{np.mean(values):.3f} ± "
        f"{np.std(values, ddof=1):.3f}"
    )


report("CER", cers)
report("WER", wers)
report("Exact", exacts)

Paper báo:

CER = x.xx ± x.xx %
WER = x.xx ± x.xx %
Exact = xx.xx ± x.xx %

Đây sẽ đáng tin hơn nhiều so với chỉ seed 42.

8. E4 — Robustness: code gần như đã có sẵn

Tin tốt là experiment.py đã có hẳn:

run_robustness()

và đã định nghĩa:

rotation:
0, 5, 10, 15, 20 degrees

noise:
0, 0.005, 0.01, 0.02, 0.05

scale:
0.8, 0.9, 1.0, 1.1, 1.2

time_warp:
0, 0.1, 0.2, 0.35

Đây chính là experiment mà chúng ta cần.

Chạy proposed model

Ví dụ:

import json
from pathlib import Path

from airwriting.experiment import run_robustness


DATA_ROOT = Path("/kaggle/input/YOUR_DATASET")

with open("manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)


run_robustness(
    checkpoint_path=Path(
        "paper_results/"
        "proposed_multiseed/"
        "seed_42/"
        "best.pt"
    ),

    data_root=DATA_ROOT,

    manifest=manifest,

    output_path=Path(
        "paper_results/"
        "robustness_proposed.json"
    ),
)

Tuy nhiên chỉ chạy proposed model chưa đủ để viết contribution về robustness.

Phải chạy ít nhất:

Spatial-only baseline
All + concat
All + gated proposed

và so sánh chúng.

9. Fig. robustness nên vẽ như thế nào?

Không gộp cả 4 loại perturbation vào một đồ thị khó đọc.

Tôi đề xuất 4 figure nhỏ hoặc 2×2 trong paper:

Rotation

x-axis:

0° 5° 10° 15° 20°

y-axis:

CER (%)

ba đường:

Spatial
Concat
Adaptive Gate

Tương tự cho Noise, Scale và Temporal Warp.

Điều chúng ta muốn chứng minh không phải chỉ:

Proposed có CER thấp nhất ở clean data.

mà là:

$$ \Delta CER_{proposed} < \Delta CER_{baseline} $$

khi perturbation tăng.

Đây mới là robustness contribution.

10. Có một chi tiết cần chỉnh trong robustness

Hiện:

settings = (
    [("rotation", value)
     for value in (0.0, 5.0, 10.0, 15.0, 20.0)]

nó chỉ quay theo hướng dương.

Nên đổi thành:

[("rotation", value)
 for value in
 (-20.0, -15.0, -10.0, -5.0,
   0.0,
   5.0, 10.0, 15.0, 20.0)]

Sau đó paper có thể plot theo:

$$ |\theta| $$

hoặc giữ signed rotation.

Tốt hơn vì handwriting/camera misalignment không có lý do chỉ theo một chiều.

11. E5 — Gate behavior analysis: experiment giá trị nhất

Đây là thực nghiệm tôi đặc biệt khuyến nghị.

Model hiện đã trả:

log_probs, lengths, fusion_weights

từ:

RecognitionModel.forward()

Với gated fusion:

fusion_weights.shape

sẽ là:

[B, T, 3]

tức là:

batch
× 128 points
× {spatial, dynamic, Fourier}

Do đó ta hoàn toàn có thể lấy:

$$ \alpha_S(t),\alpha_D(t),\alpha_F(t) $$

theo từng điểm.

12. Tạo hàm extract gate

Có thể thêm vào inference.py hoặc tạo file analysis riêng.

Ví dụ:

import torch
import numpy as np

@torch.inference_mode()
def extract_gate_weights(
    model,
    branches,
    lengths,
    device,
):
    model.eval()

    branches = {
        name: value.to(device)
        for name, value in branches.items()
    }

    lengths = lengths.to(device)

    log_probs, _, weights = model(
        branches,
        lengths,
    )

    if weights is None:
        raise ValueError(
            "Model does not use gated fusion"
        )

    weights = weights.cpu().numpy()

    return weights

Đối với một sample:

weights[0]

có dạng:

128 × 3

và:

spatial = weights[0, :, 0]
dynamic = weights[0, :, 1]
fourier = weights[0, :, 2]
13. Nên vẽ gate lên chính trajectory

Không nên chỉ plot:

weight vs sample index

mà tốt nhất là tạo hai phần:

(a) trajectory;

(b) three gate curves.

Ví dụ:

import matplotlib.pyplot as plt

t = np.arange(len(spatial))

plt.figure(figsize=(8, 4))

plt.plot(t, spatial, label="Spatial")
plt.plot(t, dynamic, label="Dynamic")
plt.plot(t, fourier, label="Fourier")

plt.xlabel("Trajectory point")
plt.ylabel("Fusion weight")

plt.ylim(0, 1)

plt.legend()
plt.tight_layout()
plt.show()

Dùng 3 case:

"bao giờ" – correct;
"ngọn núi cao vời vợi" – correct long phrase;
"bí" → "bú" – failure.

Ba sample này đã có sẵn trong ZIP và cũng chính là các qualitative examples của paper. Bản hiện tại cho biết ví dụ long phrase có gate trung bình spatial/dynamic/Fourier khoảng 0.162/0.454/0.384. 
SOICT

Bây giờ chúng ta cần đi xa hơn mean weight và xem gate thay đổi ở đâu.

14. Quantify gate specialization, không chỉ nhìn hình

Nếu chỉ có figure, reviewer có thể nói visualization mang tính anecdotal.

Ta cần thống kê trên toàn bộ test set.

Trích:

mean_gate =
weights.mean(axis=(0, 1))

cho:

Spatial
Dynamic
Fourier

Nhưng còn nên tính:

Gate entropy
$$ H_t = -\sum_b \alpha_{t,b} \log \alpha_{t,b} $$

Code:

eps = 1e-8

entropy = -np.sum(
    weights *
    np.log(weights + eps),
    axis=-1,
)

Nếu normalized:

entropy /= np.log(3)

thì:

0 = rất chuyên biệt
1 = weights gần đều nhau

Sau đó báo:

Mean fusion entropy = ...

Nếu nhỏ đáng kể hơn 1, gate thực sự đang lựa chọn representation thay vì luôn cho ba branch bằng nhau.

15. Một thống kê còn thuyết phục hơn: winning branch

Cho mỗi point:

winner = weights.argmax(axis=-1)

đếm tỷ lệ:

np.bincount(
    winner.reshape(-1),
    minlength=3,
) / winner.size

Ví dụ kết quả có thể là:

Spatial dominates : 22.4 %
Dynamic dominates : 43.1 %
Fourier dominates : 34.5 %

Đừng đoán số, hãy lấy thật từ experiment.

Bảng:

Branch selected as highest	Percentage
Spatial	xx.x%
Dynamic	xx.x%
Fourier	xx.x%

Nếu cả ba branch đều có thời điểm thắng đáng kể, đây là bằng chứng trực tiếp rằng:

representation importance varies along the trajectory.

Đây chính là novelty chúng ta muốn claim.

16. Liên hệ gate với curvature

Đây có thể thành một analysis rất đẹp.

Trong data.py, anh đã tính curvature:

curvature = numerator / (
    np.power(speed, 3) + 1e-6
)

Ta kiểm tra:

Fourier hoặc dynamic gate có tăng tại high-curvature segments hay không?

Ví dụ lấy:

curvature =
dynamic_features(points)[:, -1]

sau đó correlation:

from scipy.stats import spearmanr

rho_dynamic, p_dynamic = spearmanr(
    np.abs(curvature),
    dynamic_weight,
)

rho_fourier, p_fourier = spearmanr(
    np.abs(curvature),
    fourier_weight,
)

Report:

Spearman ρ(|κ|, αD)
Spearman ρ(|κ|, αF)

Nếu có correlation tích cực rõ thì paper có thể nói:

adaptive fusion systematically shifts toward local representations around geometrically complex trajectory regions.

Đây là contribution đẹp hơn nhiều so với chỉ CER 1.77%.

17. Vietnamese-specific error analysis

metrics.py đã làm phần lớn việc cho anh.

corpus_metrics() hiện đã trả:

top_confusions
per_character
substitutions
deletions
insertions
by_length

Bản hiện tại đã báo:

101 substitutions;
81 deletions;
25 insertions;
207 character errors / 11,721 reference characters. 
SOICT

Nhưng chúng ta cần phân loại lỗi sâu hơn.

18. Tạo nhóm tiếng Việt

Có thể viết:

TONE_MARKED = set(
    "áàảãạ"
    "ắằẳẵặ"
    "ấầẩẫậ"
    "éèẻẽẹ"
    "ếềểễệ"
    "íìỉĩị"
    "óòỏõọ"
    "ốồổỗộ"
    "ớờởỡợ"
    "úùủũụ"
    "ứừửữự"
    "ýỳỷỹỵ"
)

VIETNAMESE_SPECIAL = set(
    "ăâêôơư"
    "ĂÂÊÔƠƯ"
)

SPACE = {" "}

Tuy nhiên cách tốt hơn là dùng Unicode decomposition để không cần hard-code từng ký tự.

19. Phân biệt base vowel và diacritic

Ví dụ:

í → ú

không phải tone error vì cả hai cùng có dấu sắc.

Đó là:

base vowel substitution

Trong khi:

i → í

là:

tone/diacritic error

Ta có thể viết:

import unicodedata


def decompose_character(c):

    normalized = unicodedata.normalize(
        "NFD",
        c,
    )

    if not normalized:
        return "", ()

    base = normalized[0]

    marks = tuple(
        normalized[1:]
    )

    return base, marks

Với:

decompose_character("í")

ta sẽ được:

base = i
marks = acute

Với:

decompose_character("ị")

base vẫn là i, mark khác.

20. Error taxonomy

Dùng edit_alignment() đã có trong metrics.py.

from airwriting.metrics import edit_alignment


def classify_substitution(ref, hyp):

    ref_base, ref_marks = decompose_character(ref)
    hyp_base, hyp_marks = decompose_character(hyp)

    if ref_base == hyp_base:
        if ref_marks != hyp_marks:
            return "diacritic_or_tone"

    if ref_base != hyp_base:
        return "base_character"

    return "other"

Riêng:

if ref == " " or hyp == " ":
    return "space"

Và:

D -> deletion
I -> insertion

Sau đó trên toàn test set:

from collections import Counter

error_types = Counter()

for row in predictions:

    ref = row["reference"]
    hyp = row["prediction"]

    alignment = edit_alignment(
        list(ref),
        list(hyp),
    )

    for op, source, target in alignment:

        if op == "C":
            continue

        if op == "D":
            error_types["deletion"] += 1

        elif op == "I":
            error_types["insertion"] += 1

        elif source == " " or target == " ":
            error_types["space"] += 1

        else:
            error_types[
                classify_substitution(
                    source,
                    target,
                )
            ] += 1
21. Table IV nên báo như sau
Vietnamese orthographic error analysis
Error type	Count	% of errors
Base-letter substitution		
Tone/diacritic substitution		
Space error		
Deletion		
Insertion		

Có thể thêm Top 10 confusions:

Reference	Prediction	Count
...	...	...

Trong paper hiện tại bí → bú đã được phân tích như một base-vowel substitution và cho thấy các chuyển động phân biệt nguyên âm rất ngắn. 
SOICT Phân tích toàn test set sẽ biến observation đó thành quantitative evidence.

22. E6 — Tôi rất khuyến nghị: lexical-disjoint evaluation

Đây là experiment có thể tạo khác biệt lớn cho bài.

Tôi thấy trong manifest.py đã có sẵn:

create_label_disjoint_manifest()

và code còn kiểm tra:

evaluation_protocol ==
"lexical-disjoint open-phrase recognition"

Điều này rất hay.

Nó nghĩa là:

training phrases
∩
test phrases
=
∅

nhưng các character trong test vẫn từng xuất hiện trong train.

Ví dụ:

Training:
"xin chào"
"cảm ơn"

Test:
"xin cảm ơn"

Ở đây phrase test chưa từng thấy nhưng các ký tự đã biết.

Đây mới thật sự kiểm tra:

$$ \boxed{ \text{trajectory} \rightarrow \text{character sequence} } $$

chứ không phải ghi nhớ 660 labels.

23. Tạo lexical-disjoint manifest

Ví dụ:

from pathlib import Path

from airwriting.manifest import (
    create_label_disjoint_manifest
)

manifest = create_label_disjoint_manifest(
    data_root=Path("/path/to/VNI"),
    output_path=Path(
        "manifest_lexical_disjoint.json"
    ),
    seed=20260908,
)

Sau đó train:

config = ExperimentConfig(
    name="lexical_disjoint_proposed",

    representation="all",
    fusion="gated",

    backbone="conformer",
    positional="relative",

    fourier_scales=2,

    length=128,

    augment=True,

    seed=42,
)

run_experiment(
    config,
    DATA_ROOT,
    manifest,
    OUTPUT_ROOT,
)

Đây sẽ khó hơn rất nhiều so với source-disjoint evaluation hiện tại.

Không nên kỳ vọng CER vẫn 1.77%.

Nếu CER tăng lên chẳng hạn 10–20%, cũng không phải thất bại. Điều quan trọng là so với baseline dưới cùng protocol.

24. Lexical-disjoint nên so sánh ít model

Không cần chạy hết 20 cấu hình.

Chỉ:

Transformer / Spatial
Conformer / Spatial
Conformer / All + gated

hoặc thậm chí:

Conformer spatial
Proposed

Nếu proposed tốt hơn rõ khi phrase hoàn toàn unseen thì contribution sequence transcription rất mạnh.

25. Một experiment tôi muốn thêm vì phát hiện từ code

Đây là vấn đề quan trọng nhất về mặt methodology.

Pipeline hiện tại:

Raw trajectory
↓
smooth
↓
spline
↓
normalize
↓
arc-length resample
↓
dynamic_features

Trong khi paper mô tả dynamic branch gồm:

speed
acceleration
direction
curvature

SOICT

Nhưng arc-length resampling cố tình làm khoảng cách dọc theo đường cong tương đối đồng đều.

Do đó:

$$ v_t=\|\Delta p_t\| $$

sau bước này không còn là writing speed gốc.

Đây là điểm reviewer giỏi signal processing/time-series có thể nhận ra.

26. Có hai cách xử lý

Tôi nghiêng về cách A cho paper này.

A. Không thay pipeline chính

Giữ checkpoint hiện tại để không phá toàn bộ result.

Nhưng đổi terminology:

Thay:

motion dynamics / writing speed

bằng:

local trajectory descriptors

hoặc:

local geometric-motion descriptors

Và mô tả:

first differences,
local displacement magnitude,
unit tangent direction,
displacement variation,
and curvature

Cách này an toàn.

B. Làm thêm temporal-resampling experiment

Đây sẽ khoa học hơn nhưng phải train lại.

Thay arc-length resampling bằng:

def temporal_resample(
    points,
    length,
):

    source_t = np.linspace(
        0.0,
        1.0,
        len(points),
    )

    target_t = np.linspace(
        0.0,
        1.0,
        length,
    )

    return np.column_stack([
        np.interp(
            target_t,
            source_t,
            points[:, axis],
        )
        for axis in range(2)
    ])

Lúc đó local displacement sẽ phản ánh tốc độ tương đối tốt hơn.

So sánh:

Resampling	CER
Temporal/index	
Arc-length	

Tuy nhiên tôi coi đây là optional, không phải experiment đầu tiên phải chạy.

27. Thứ tự chạy tối ưu để tiết kiệm GPU

Tôi đề xuất anh chạy đúng thứ tự sau.

Đợt 1 — seed 42:

E1:
7 representation models

E2:
4 fusion models

Nhưng lưu ý:

all + gated

trùng giữa hai experiment nên tổng thực tế khoảng 10 runs, không phải 11.

Sau đó kiểm tra kết quả.

Nếu gated không có dấu hiệu tốt hơn concat/static, dừng lại và chúng ta phải chỉnh lại novelty trước khi tốn GPU thêm.

Nếu kết quả hợp lý:

Đợt 2:

3 seeds ×
3 important models

≈ 9 runs, trong đó seed 42 đã có nên chỉ cần thêm khoảng 6.

Đợt 3:

robustness

không train lại.

Đợt 4:

gate analysis
error analysis

không train lại.

Đợt 5 – optional nhưng rất giá trị:

lexical-disjoint:
2–3 models

Như vậy tổng GPU workload vẫn hợp lý.

28. Cuối cùng bài nên có 4 bảng chính

Tôi sẽ cấu trúc Results như sau.

Table 1 — Backbone comparison

Giữ gần giống bản hiện tại:

BiLSTM
GRU
Transformer
Conformer

Bản hiện tại đã cho thấy Conformer 1.85% CER so với Transformer 11.79%, là một chênh lệch rất lớn. 
SOICT

Table 2 — Representation ablation
S
D
F
SD
SF
DF
SDF
Table 3 — Fusion ablation
mean
concat
static
adaptive gate

kèm:

mean ± std

cho ba model chính nếu đủ chỗ.

Table 4 — Error / gate analysis

Vietnamese-specific errors + gate statistics.

29. Và 3 figure quan trọng

Fig. 1 — Proposed architecture

Giữ kiến trúc hiện tại nhưng đổi framing sang:

Representation-adaptive trajectory modeling.

Fig. 2 — Gate behavior

Một trajectory + ba curves:

$$ \alpha_S(t),\alpha_D(t),\alpha_F(t). $$

Đây có thể trở thành figure quan trọng nhất của contribution.

Fig. 3 — Robustness

CER vs:

rotation
noise
time warp

không nhất thiết cần scale nếu giới hạn trang.

30. Tiêu chí để quyết định paper có đủ mạnh hay không

Sau khi chạy, tôi sẽ nhìn trước hết vào ba câu hỏi sau:

Điều kiện A

Có ít nhất một trong:

$$ CER_{SDF} < CER_S $$

và:

$$ CER_{SDF} < CER_{SD/SF/DF}. $$

Nếu có → multi-representation được chứng minh.

Điều kiện B
$$ CER_{gated} < CER_{concat/static} $$

hoặc gated cho robustness tốt hơn rõ.

Nếu có → adaptive fusion novelty được chứng minh.

Điều kiện C

Gate analysis cho thấy:

$$ \alpha_S(t), \alpha_D(t), \alpha_F(t) $$

thay đổi đáng kể theo trajectory, thay vì gần như cố định.

Nếu có → claim representation-adaptive thực sự có bằng chứng.

Nếu cả A+B+C đều đạt, tôi đánh giá câu chuyện paper sẽ mạnh hơn bản hiện tại rất đáng kể.

Việc nên làm ngay

Tôi đề nghị chưa chạy lexical-disjoint và chưa sửa preprocessing. Trước tiên chỉ chạy E1 + E2 ở seed 42, vì khoảng 10 runs này sẽ cho chúng ta biết novelty “adaptive multi-representation” có thực sự được dữ liệu ủng hộ hay không.

Sau khi anh có thư mục paper_results với result.json của các run đó, gửi ZIP kết quả cho tôi. Tôi có thể từ các file JSON tự động tổng hợp toàn bộ Table II–III, kiểm tra statistical trend, chọn các cấu hình cần multi-seed, rồi hướng dẫn chính xác bước robustness/gate analysis tiếp theo.

