"""Frozen official-source identity for the four-layer GLM-5.2 BF16 pilot.

These values were cross-checked against the independent R10 BF16 source
inventory.  They are deliberately embedded here: source validation must not
depend on mutable metadata from the local quantized checkpoint or on a caller
supplied hash list.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


SOURCE_REPO: Final = "zai-org/GLM-5.2"
SOURCE_REVISION: Final = "b4734de4facf877f85769a911abafc5283eab3d9"

EXPECTED_INDEX_FILENAME: Final = "model.safetensors.index.json"
EXPECTED_INDEX_BYTES: Final = 5_408_032
EXPECTED_INDEX_SHA256: Final = (
    "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
)
EXPECTED_INDEX_TOTAL_BYTES: Final = 1_506_659_919_872
EXPECTED_INDEX_TENSOR_COUNT: Final = 59_585

EXPECTED_CONFIG_FILENAME: Final = "config.json"
EXPECTED_CONFIG_SHA256: Final = (
    "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
)

EXPECTED_TOTAL_SHARD_BYTES: Final = 96_536_706_248


@dataclass(frozen=True)
class ShardIdentity:
    """Cryptographic and structural identity of one official source shard."""

    bytes: int
    sha256: str
    header_sha256: str
    tensor_count: int


EXPECTED_SHARDS: Final[Mapping[str, ShardIdentity]] = MappingProxyType(
    {
        "model-00068-of-00282.safetensors": ShardIdentity(
            5_366_406_928,
            "70b55097b967324aeb90b6bd84c6ceed6cac8f315f24e37227dde79e2cd88905",
            "6457936464d7bd3ba8cface7cc044d29ffa78eadb8df2ebca0913c78b7fda199",
            211,
        ),
        "model-00069-of-00282.safetensors": ShardIdentity(
            5_360_347_320,
            "29b77893e10c09132b082da8adeb14aa202430c197283247def1a52f5a8b8a6c",
            "7d1298e6ef87c389c862ae5e265b95e8fc4d190648b655e57f4428552dc97f8b",
            213,
        ),
        "model-00070-of-00282.safetensors": ShardIdentity(
            5_360_347_296,
            "8b34067e0fed2016bf41648c4267ff413457c464ac13285a860bf4fd846eca60",
            "7ac5ff0f7880c0bb481bc6d37e4e9996dd3806d95acdaa60c017531abf15497b",
            213,
        ),
        "model-00071-of-00282.safetensors": ShardIdentity(
            5_360_347_112,
            "2e5e1db8a81b690febb6c412746fb816ebc1ca92cdcfeabb74c42210401bd9ee",
            "f9d9d719df9cd9ed6dcb0d4ddac7b4b357ca101e1d3bb4cf10ee6af5122d60d7",
            213,
        ),
        "model-00072-of-00282.safetensors": ShardIdentity(
            5_366_407_000,
            "7214d454a5541e25239e1323d97d1b01a0557c010676a4496523828785ebe377",
            "d186543ca22be5c047072aa10d663b9a748a1c715cde3ed8d23db0e0dc149615",
            211,
        ),
        "model-00167-of-00282.safetensors": ShardIdentity(
            5_366_406_840,
            "5e3273b1fa82ee849e4e494590311f0b15bbbcf89963aaf81ba21a29063c75df",
            "e09314791d09637d4f02cb28bf01a20d065ad8c93be91a8355d0ff58a91af40a",
            211,
        ),
        "model-00168-of-00282.safetensors": ShardIdentity(
            5_360_347_320,
            "5cad87989d84fb80d514cfd4b87f9eded095e82b0e95d7136bc67425ff0ee5e7",
            "eed1e124a0753f5ba7c2b99741ed04ed052005ace8978db6e57f48e5f50fc99a",
            213,
        ),
        "model-00169-of-00282.safetensors": ShardIdentity(
            5_360_347_312,
            "1cee36a399dd44c4e2110ad21596569107fe949ae826d19d7c14823b97fecab3",
            "f9319b6db85da1d76870711048a4c6085406d168ea4ae7120c147e71d45b0e5f",
            213,
        ),
        "model-00170-of-00282.safetensors": ShardIdentity(
            5_360_347_184,
            "8e41ad50549a8a94cc584310509487d2545a6ebdb3164f8d4a182fff9c7881b4",
            "c5d63baabd73ad31a05f7357818f78b33ff4040027bcdfcf3d5d65bf3a6166fd",
            213,
        ),
        "model-00171-of-00282.safetensors": ShardIdentity(
            5_366_406_904,
            "484f4f181fee543edba881406f591857edf8beffc0d7a99abb63442ae62ab81e",
            "b45504614184fbd514ed68e34d687b3909f49d2684a90e33dcc0b73de966c7a3",
            211,
        ),
        "model-00197-of-00282.safetensors": ShardIdentity(
            5_366_406_808,
            "f7297ce4da1aebf42024bbbd6f562c1e29f91d0907ac986ce118e0f6f51ce5b9",
            "d86f30ac969caab104a177a04d422f17573b8738dd0de47d50021f0467158946",
            211,
        ),
        "model-00198-of-00282.safetensors": ShardIdentity(
            5_360_347_104,
            "6bad23a379cea51d80fbe39ef8173350f3b2e05052dd8b46d93d31cc686d21ca",
            "ee3e3cd776af7b0945c0603df66b1f5acac9bc07deec9d7141023f29f95e334e",
            213,
        ),
        "model-00199-of-00282.safetensors": ShardIdentity(
            5_360_347_064,
            "b8010fc4543568613fb750634bd6bd095783f9e5b59924d79a103b4c1d854d60",
            "1fed79a289820a55a3d037bfda6d9f2577d4543d51528b76634ba382574cfa3b",
            213,
        ),
        "model-00200-of-00282.safetensors": ShardIdentity(
            5_368_361_544,
            "8d058b156900395078dd8350f24c16d492d9854d54d019d357bb583bd9a23d3a",
            "7f818db366d4bb67ad3835a8edaf07cc97e6a8374e1fae526b5f46638c4ab753",
            216,
        ),
        "model-00267-of-00282.safetensors": ShardIdentity(
            5_366_406_960,
            "72531de28c8221463eea9f82c1e2c6bc6fb6e2b5c89c21e3237dc707516bc7ba",
            "afcc5c084d101eb36b862f85d0201dc1421d69031f350f575aad8368b1afb468",
            211,
        ),
        "model-00268-of-00282.safetensors": ShardIdentity(
            5_360_347_320,
            "db6bfce751a1a56ae17856631552fb4f83821c300822a909da48267f5d82a976",
            "0a10e5c4a05a05245a157da3409a5ca3b0fbbfbea9187d717939aa3bb6855c77",
            213,
        ),
        "model-00269-of-00282.safetensors": ShardIdentity(
            5_360_347_264,
            "49d6a9e4c256bc01d18fac9dd8d68177f1852a4b0e9d5f7b8d7971f9a3075652",
            "e887936845a6f77fd16263603dcd4f40236ff2aeaa38e4009388ba350b1c66f1",
            213,
        ),
        "model-00270-of-00282.safetensors": ShardIdentity(
            5_366_430_968,
            "d74106256f061e73000e9660d157bd22254d2a5692cf9466d76dfea6985c0924",
            "0d185dfb01c6bfb3089f087bcbf0ff0b67fa979cd36190baa3bea1e83da5ac5b",
            208,
        ),
    }
)


if len(EXPECTED_SHARDS) != 18:  # pragma: no cover - import-time invariant
    raise AssertionError("the BF16 pilot manifest must contain exactly 18 shards")
if sum(item.bytes for item in EXPECTED_SHARDS.values()) != EXPECTED_TOTAL_SHARD_BYTES:
    raise AssertionError("the BF16 pilot manifest has an invalid byte total")
