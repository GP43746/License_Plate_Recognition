import easyocr


class OCREngine:
    def __init__(self, use_gpu=True):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def read(self, image):
        """
        Returns (text: str, confidence: float).
        text is cleaned alphanumeric uppercase string.
        confidence is the average over all detected words (0.0 if none).
        """
        if image is None:
            return "", 0.0

        results = self.reader.readtext(image)

        if not results:
            return "", 0.0

        # results: list of (bbox, text, confidence)
        raw_text = ""
        total_conf = 0.0
        count = 0

        for _, word, conf in results:
            raw_text += word
            total_conf += conf
            count += 1

        avg_conf = round(total_conf / count, 3) if count > 0 else 0.0

        cleaned = raw_text.upper().replace(" ", "")
        cleaned = ''.join(c for c in cleaned if c.isalnum())

        return cleaned, avg_conf