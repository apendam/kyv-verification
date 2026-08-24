"""AWS Rekognition VRN detector — copied (near-verbatim) from
truck_verification_pipeline/step14_rekognition_detector.py. Real text detection +
Indian-plate parsing (two-line handling, brand/slogan rejection, O/I fuzzy-correction),
replacing Claude entirely for Q2's plate reading.

The YOLO-fallback path from the original module is dropped here (that pipeline kept it
for backward-compat call sites only — its own README already says "YOLO and the
positional/bottom-centre fallbacks have been intentionally removed... Rekognition is the
single source of truth"). Vehicle-type detection lives separately in backends/vehicle.py
(Q1), not here.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import boto3
from PIL import Image

from vfiv import config


class RekognitionCredentialError(RuntimeError):
    """Raised when AWS credentials are missing, invalid, or expired."""


# ── Indian VRN pattern matcher ────────────────────────────────────────────────
_VRN_PATTERNS = [
    re.compile(r'\b([A-Z]{2})\s*(\d{1,3})\s*([A-Z]{1,3})\s*(\d{1,4})\b'),  # standard + AP
    re.compile(r'\b(\d{2})\s*(BH)\s*(\d{1,4})\s*([A-Z]{1,2})\b'),           # BH series
    re.compile(r'\b([A-Z]{2})\s*(\d{2})\s*(\d{4})\b'),                      # seriesless: WB 63 1486
]

# OCR-confusion fallback. Indian RTOs deliberately never use O or I in registration
# numbers (too easily confused with 0/1) — coerce O->0, I->1 and re-run the strict
# patterns. Other look-alikes (S/B/G/Z/D/L) are left alone since those ARE valid
# series letters.
_OI_TO_DIGIT = str.maketrans("OI", "01")

_VALID_STATES = frozenset((
    "AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ "
    "NL OD OR PB PY RJ SK TN TR TS UK UA UP WB"
).split())

# Decorative/slogan text painted on Indian trucks — must never be mistaken for a VRN.
_DECOR_WORDS = {
    "GOOD", "GOODS", "CARRIER", "CARRIERS", "CARRIAR", "PUBLIC", "PRIVATE", "NATIONAL",
    "PERMIT", "AIP", "ALL", "INDIA", "BHARAT", "TRANSPORT", "ROADWAYS", "ROADLINES", "LOGISTICS",
    "CARGO", "MOVERS", "PACKERS", "FREIGHT", "TEMPO", "TRAVELS", "TRAVEL", "COURIER",
    "CARRYING", "LOAD", "CAPACITY",
    "ROAD", "KING", "HIGHWAY", "STAR", "EMPEROR", "BADSHAH", "DON", "RAJA", "SARTAJ",
    "SHAHENSHAH", "SIKANDAR", "SARDAR", "MAHARAJA", "NAWAB", "SULTAN", "ROYAL", "SHER",
    "SHERE", "TIGER", "LION", "CHAMP", "CHAMPION", "BOSS", "LUCKY", "LUCK", "SHAAN", "SHAN",
    "HORN", "PLEASE", "BLOW", "USE", "DIPPER", "NIGHT", "WAIT", "SIDE", "STOP", "KEEP",
    "DISTANCE", "DRIVE", "SAFE", "SAFETY", "SLOW", "AHEAD", "LEFT", "RIGHT", "TURN",
    "BRAKE", "DIP", "SOUND", "OKTATA",
    "JAI", "MATA", "MATADI", "MATARANI", "MAA", "RANI", "HIND", "BHOLE", "BHOLENATH",
    "OM", "AUM", "SHRI", "RAM", "BAJRANG", "BAJRANGBALI", "HANUMAN", "BALAJI", "SAI",
    "BABA", "SAIBABA", "WAHE", "GURU", "WAHEGURU", "GOD", "BLESS", "BLESSING", "BLESSINGS",
    "MAHAKAL", "MAHADEV", "SHIV", "SHIVA", "SHAKTI", "DURGA", "KALI", "LAXMI", "LAKSHMI",
    "MAHALAXMI", "GANESH", "GANPATI", "GANESHA", "KRISHNA", "RADHE", "RADHA", "ALLAH",
    "BISMILLAH", "MASHALLAH", "KHUDA", "NABI", "GANGA", "GAYATRI", "SANTOSHI", "VAISHNO",
    "DEVI", "BHAVANI", "ANNAPURNA", "TIRUPATI", "AYYAPPA", "MURUGAN", "AMMA", "JESUS",
    "CHRIST", "MOTHER", "FATHER", "JWALA", "KHATU", "SHYAM", "NAMAH", "NAMAHA", "SWAMI",
    "MARUTI", "NANDI",
    "GREETING", "GREETINGS", "WELCOME", "MOM", "DAD", "MUMMY", "PAPA", "DADA", "NANA",
    "BHAI", "BROTHERS", "SONS", "FAMILY", "MEMORY", "LOVE", "MAMTA", "ASHIRWAD",
    "ASHIRVAD", "DUA", "BETA",
    "DRIVER", "OWNER", "MALIK", "CONTACT", "MOBILE", "PHONE", "CALL", "DIESEL", "ONLY",
    "MODEL", "TAX", "FITNESS", "INSURANCE", "CHASSIS", "ENGINE", "SLEEPER", "DELUXE",
    "EXPRESS", "SUPER", "TURBO", "POWER", "WEIGHT", "TONS", "SELF", "HYDRAULIC",
    "ASHOK", "LEYLAND", "TATA", "EICHER", "MAHINDRA", "VOLVO", "SCANIA", "BHARATBENZ",
    "FORCE", "DAIMLER", "ISUZU", "MERCEDES", "BENZ", "MAN",
    "PUNJAB", "BIHAR", "RAJASTHAN", "GUJARAT", "HARYANA", "KERALA", "ASSAM", "ODISHA",
    "BENGAL", "MUMBAI", "DELHI", "CHENNAI", "KOLKATA", "INDORE", "NAGPUR", "SURAT",
}
_DECOR_WB_RE = re.compile(r"\b(" + "|".join(sorted(_DECOR_WORDS, key=len, reverse=True)) + r")\b")
_DECOR_SUB_RE = re.compile("(" + "|".join(sorted((w for w in _DECOR_WORDS if len(w) >= 4),
                                                  key=len, reverse=True)) + ")")


def _fuzzy_vrn(candidate: str) -> str | None:
    coerced = candidate.translate(_OI_TO_DIGIT)
    if coerced == candidate:
        return None
    for pat in _VRN_PATTERNS:
        m = pat.search(coerced)
        if m:
            vrn = "".join(m.groups())
            if vrn[:2].isalpha() and vrn[:2] not in _VALID_STATES:
                continue
            return vrn
    return None


def _parse_vrn(text_lines: list[str]) -> str | None:
    raw = " ".join(text_lines).upper()
    raw = re.sub(r"\bIND(IA)?\b", " ", raw)
    raw = re.sub(r"(?<=\d)IND(?=\d)", "", raw)
    raw = re.sub(r"[^A-Z0-9]+", " ", raw).strip()
    raw = _DECOR_SUB_RE.sub(" ", raw)
    raw = _DECOR_WB_RE.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    nospace = re.sub(r"\s+", "", raw)
    for candidate in (raw, nospace):
        for pat in _VRN_PATTERNS:
            m = pat.search(candidate)
            if m:
                return "".join(m.groups())
    for candidate in (raw, nospace):
        v = _fuzzy_vrn(candidate)
        if v:
            return v
    return None


_MAKE_CANON = {
    "ASHOK": "ASHOK LEYLAND", "LEYLAND": "ASHOK LEYLAND", "ASHOK LEYLAND": "ASHOK LEYLAND",
    "TATA": "TATA", "EICHER": "EICHER", "MAHINDRA": "MAHINDRA", "VOLVO": "VOLVO",
    "BHARATBENZ": "BHARATBENZ", "BHARAT BENZ": "BHARATBENZ", "MAN": "MAN",
    "SML": "SML ISUZU", "ISUZU": "ISUZU", "DAIMLER": "DAIMLER", "MERCEDES": "MERCEDES",
    "SCANIA": "SCANIA", "DAF": "DAF", "IVECO": "IVECO", "RENAULT": "RENAULT", "FORCE": "FORCE",
}


def _canon_make(brand: str) -> str:
    return _MAKE_CANON.get(brand.upper().strip(), brand.upper().strip())


def _merge_bb(a: dict, b: dict) -> dict:
    left = min(a["Left"], b["Left"])
    top = min(a["Top"], b["Top"])
    right = max(a["Left"] + a["Width"], b["Left"] + b["Width"])
    bottom = max(a["Top"] + a["Height"], b["Top"] + b["Height"])
    return {"Left": left, "Top": top, "Width": right - left, "Height": bottom - top}


def _is_stacked(a: dict, b: dict) -> bool:
    ax2, bx2 = a["Left"] + a["Width"], b["Left"] + b["Width"]
    overlap = min(ax2, bx2) - max(a["Left"], b["Left"])
    if overlap < 0.4 * min(a["Width"], b["Width"]):
        return False
    gap = b["Top"] - (a["Top"] + a["Height"])
    return -0.5 * a["Height"] <= gap <= 1.2 * a["Height"]


def _is_beside(a: dict, b: dict) -> bool:
    ay2, by2 = a["Top"] + a["Height"], b["Top"] + b["Height"]
    overlap = min(ay2, by2) - max(a["Top"], b["Top"])
    if overlap < 0.4 * min(a["Height"], b["Height"]):
        return False
    gap = b["Left"] - (a["Left"] + a["Width"])
    return -0.5 * a["Width"] <= gap <= 1.2 * a["Width"]


def _is_adjacent(a: dict, b: dict) -> bool:
    return (_is_stacked(a, b) or _is_stacked(b, a)
            or _is_beside(a, b) or _is_beside(b, a))


@dataclass
class PlateDetection:
    crop: Image.Image
    bbox: list[int]
    confidence: float
    vrn: str | None
    source: str
    truck_bbox: list[int] | None = None
    make: str | None = None
    vehicle_class: str | None = None


class RekognitionPlateDetector:
    """One instance per process — boto3 client is reused."""

    def __init__(self, region: str | None = None):
        self.client = boto3.client("rekognition", region_name=region or config.AWS_REKOGNITION_REGION)

    def _image_bytes(self, image, max_bytes: int = 4_900_000) -> bytes:
        img = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
        scale = (max_bytes / len(data)) ** 0.5
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    _BRAND_BLACKLIST = {
        "ASHOK LEYLAND", "ASHOK", "LEYLAND",
        "TATA", "EICHER", "MAHINDRA", "VOLVO", "MAN",
        "BHARATBENZ", "FORCE", "SML", "ISUZU", "DAIMLER",
        "MERCEDES", "SCANIA", "DAF", "IVECO", "RENAULT",
    }

    def _is_brand_text(self, text: str) -> bool:
        upper = text.upper().strip()
        return any(brand in upper for brand in self._BRAND_BLACKLIST)

    _TRUCK_LABELS = {
        "Truck", "Lorry", "Semi Trailer Truck", "Trailer Truck",
        "Pickup Truck", "Tow Truck", "Moving Van", "Trailer", "Bus",
        "Tractor", "Dump Truck", "Cement Truck", "Tanker",
    }
    _CAR_LABELS = {
        "Car", "Sedan", "Sports Car", "Coupe", "Suv", "Hatchback",
        "Convertible", "Limo", "Race Car", "Antique Car", "Jaguar Car",
    }
    _GENERIC_VEHICLE_LABELS = {
        "Vehicle", "Transportation", "Automobile", "Van", "Minivan",
        "Machine", "Motor Vehicle",
    }

    def detect(self, image, text_conf: float = 45.0) -> PlateDetection | None:
        """Strategy A: VRN-first via detect_text (preferred). Strategy B: label
        fallback via detect_labels' "License Plate" object. Strategy C: on-truck
        text present but unparseable -> surfaced (vrn=None) for human review rather
        than silently "no plate"."""
        image_pil = (image if isinstance(image, Image.Image) else Image.open(image)).convert("RGB")
        img_bytes = self._image_bytes(image_pil)
        w, h = image_pil.size

        text_resp = self.client.detect_text(Image={"Bytes": img_bytes})

        lines = []
        for det in text_resp.get("TextDetections", []):
            if det["Type"] != "LINE" or det["Confidence"] < text_conf:
                continue
            text = det["DetectedText"].upper().strip()
            if self._is_brand_text(text):
                continue
            lines.append({"text": text, "bb": det["Geometry"]["BoundingBox"], "conf": det["Confidence"]})

        label_resp = self.client.detect_labels(
            Image={"Bytes": img_bytes}, MinConfidence=60.0, Features=["GENERAL_LABELS"])
        truck_boxes, car_boxes, vehicle_boxes = [], [], []
        _vehicle_set = self._TRUCK_LABELS | self._CAR_LABELS | self._GENERIC_VEHICLE_LABELS
        for lab in label_resp.get("Labels", []):
            if lab["Name"] in self._TRUCK_LABELS:
                truck_boxes.extend(inst["BoundingBox"] for inst in lab.get("Instances", []))
            if lab["Name"] in self._CAR_LABELS:
                car_boxes.extend(inst["BoundingBox"] for inst in lab.get("Instances", []))
            if lab["Name"] in _vehicle_set:
                vehicle_boxes.extend(inst["BoundingBox"] for inst in lab.get("Instances", []))

        make = None
        for tdet in text_resp.get("TextDetections", []):
            if tdet["Type"] != "LINE" or tdet["Confidence"] < text_conf:
                continue
            up = tdet["DetectedText"].upper().strip()
            hit = next((b for b in self._BRAND_BLACKLIST if b in up), None)
            if hit:
                make = _canon_make(hit)
                break
        vehicle_class = next(
            (lab["Name"] for lab in label_resp.get("Labels", [])
             if lab["Name"] in (self._TRUCK_LABELS | self._CAR_LABELS)), None)

        def _vehicle_box_for(plate_px):
            if not vehicle_boxes or plate_px is None:
                return None
            pcx, pcy = (plate_px[0] + plate_px[2]) / 2, (plate_px[1] + plate_px[3]) / 2
            containing = []
            for vb in vehicle_boxes:
                x1, y1 = vb["Left"] * w, vb["Top"] * h
                x2, y2 = (vb["Left"] + vb["Width"]) * w, (vb["Top"] + vb["Height"]) * h
                if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                    containing.append(([int(x1), int(y1), int(x2), int(y2)], (x2 - x1) * (y2 - y1)))
            return max(containing, key=lambda t: t[1])[0] if containing else None

        def _center(bb):
            return (bb["Left"] + bb["Width"] / 2, bb["Top"] + bb["Height"] / 2)

        def _inside(pt, box):
            return (box["Left"] <= pt[0] <= box["Left"] + box["Width"] and
                    box["Top"] <= pt[1] <= box["Top"] + box["Height"])

        def _on_truck(bb):
            return any(_inside(_center(bb), t) for t in truck_boxes)

        def _on_car(bb):
            return any(_inside(_center(bb), c) for c in car_boxes)

        candidates = []
        for ln in lines:
            vrn = _parse_vrn([ln["text"]])
            if vrn:
                candidates.append((vrn, ln["bb"], ln["conf"]))

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if not _is_adjacent(a["bb"], b["bb"]):
                    continue
                for first, second in ((a, b), (b, a)):
                    vrn = _parse_vrn([first["text"], second["text"]])
                    if vrn:
                        candidates.append((vrn, _merge_bb(a["bb"], b["bb"]), min(a["conf"], b["conf"])))
                        break

        if truck_boxes:
            candidates = [c for c in candidates if not (_on_car(c[1]) and not _on_truck(c[1]))]

        if candidates:
            candidates.sort(key=lambda c: (_on_truck(c[1]), c[2]), reverse=True)
            vrn, bb, conf = candidates[0]
            pad_x = int(bb["Width"] * w * 0.15)
            pad_y = int(bb["Height"] * h * 0.40)
            x1 = max(0, int(bb["Left"] * w) - pad_x)
            y1 = max(0, int(bb["Top"] * h) - pad_y)
            x2 = min(w, int((bb["Left"] + bb["Width"]) * w) + pad_x)
            y2 = min(h, int((bb["Top"] + bb["Height"]) * h) + pad_y)
            crop = image_pil.crop((x1, y1, x2, y2))
            return PlateDetection(
                crop=crop, bbox=[x1, y1, x2, y2], confidence=conf / 100.0, vrn=vrn,
                source="rekognition(text-match)", truck_bbox=_vehicle_box_for([x1, y1, x2, y2]),
                make=make, vehicle_class=vehicle_class,
            )

        plate_instances = []
        for label in label_resp.get("Labels", []):
            if label["Name"].lower() in ("license plate", "licence plate", "number plate"):
                for inst in label.get("Instances", []):
                    plate_instances.append((inst["BoundingBox"], inst["Confidence"]))

        for bb, conf in sorted(plate_instances, key=lambda x: x[1], reverse=True):
            pad = 6
            x1 = max(0, int(bb["Left"] * w) - pad)
            y1 = max(0, int(bb["Top"] * h) - pad)
            x2 = min(w, int((bb["Left"] + bb["Width"]) * w) + pad)
            y2 = min(h, int((bb["Top"] + bb["Height"]) * h) + pad)
            crop = image_pil.crop((x1, y1, x2, y2))
            crop_buf = io.BytesIO()
            crop.save(crop_buf, format="JPEG", quality=95)
            val_resp = self.client.detect_text(Image={"Bytes": crop_buf.getvalue()})
            crop_lines = [d["DetectedText"] for d in val_resp.get("TextDetections", [])
                          if d["Type"] == "LINE" and d["Confidence"] >= text_conf]
            if any(self._is_brand_text(t) for t in crop_lines):
                continue
            vrn = _parse_vrn(crop_lines)
            return PlateDetection(
                crop=crop, bbox=[x1, y1, x2, y2], confidence=conf / 100.0, vrn=vrn,
                source="rekognition(label)", truck_bbox=_vehicle_box_for([x1, y1, x2, y2]),
                make=make, vehicle_class=vehicle_class,
            )

        if truck_boxes:
            on_truck_lines = [ln for ln in lines if _on_truck(ln["bb"])
                              and any(ch.isdigit() for ch in ln["text"])]
            if on_truck_lines:
                bbs = [ln["bb"] for ln in on_truck_lines]
                merged = bbs[0]
                for extra in bbs[1:]:
                    merged = _merge_bb(merged, extra)
                pad_x = int(merged["Width"] * w * 0.15)
                pad_y = int(merged["Height"] * h * 0.40)
                x1 = max(0, int(merged["Left"] * w) - pad_x)
                y1 = max(0, int(merged["Top"] * h) - pad_y)
                x2 = min(w, int((merged["Left"] + merged["Width"]) * w) + pad_x)
                y2 = min(h, int((merged["Top"] + merged["Height"]) * h) + pad_y)
                return PlateDetection(
                    crop=image_pil.crop((x1, y1, x2, y2)), bbox=[x1, y1, x2, y2],
                    confidence=0.0, vrn=None, source="rekognition(unparsed-text)",
                    truck_bbox=_vehicle_box_for([x1, y1, x2, y2]), make=make,
                    vehicle_class=vehicle_class,
                )

        return None


def detect_plate(image, rekognition: RekognitionPlateDetector | None = None) -> PlateDetection | None:
    """Raises RekognitionCredentialError on invalid/expired AWS credentials — surfaced
    loudly rather than silently degrading. Returns None if nothing was found."""
    rekognition = rekognition or RekognitionPlateDetector()
    try:
        return rekognition.detect(image)
    except Exception as e:
        msg = str(e)
        if any(tok in msg for tok in (
            "InvalidClientTokenId", "ExpiredToken", "UnrecognizedClientException",
            "security token", "AccessDenied", "credentials",
        )):
            raise RekognitionCredentialError(
                "AWS Rekognition credentials are invalid or expired. "
                "Refresh AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.\n"
                f"(underlying error: {msg})"
            ) from e
        raise
