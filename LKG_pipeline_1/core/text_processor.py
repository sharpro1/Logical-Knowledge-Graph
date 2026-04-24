import re
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    _instance = None
    _nlp = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextProcessor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._nlp = None
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                logger.info("Loaded spaCy model 'en_core_web_sm' for lemmatization.")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Falling back to simple normalization.")
        except Exception as e:
            logger.warning(f"Failed to initialize spaCy: {str(e)}. Falling back to simple normalization.")

    def normalize(self, text: str) -> str:
        """
        标准文本规范化：
        1. 转小写
        2. 去除首尾空格
        3. 规范化空白字符（将多个空格/换行转为单个空格）
        """
        if not text:
            return ""
        
        # 1. Lowercase and strip
        text = text.lower().strip()
        
        # 2. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def lemmatize(self, text: str) -> str:
        """
        词形还原：将单词还原为词根形式。
        """
        normalized_text = self.normalize(text)
        
        if not self._nlp:
            return normalized_text
            
        doc = self._nlp(normalized_text)
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas)

    def get_canonical_id_str(self, name: str, type_str: str) -> str:
        """
        生成用于 Hash ID 的规范化字符串。
        Format: "type:lemma_name"
        """
        lemma_name = self.lemmatize(name)
        norm_type = self.normalize(type_str)
        return f"{norm_type}:{lemma_name}"

# Global instance
processor = TextProcessor()

