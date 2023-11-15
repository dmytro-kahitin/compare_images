import re
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config.environment_manager import EnvironmentManager


class ImageSimilarityService(EnvironmentManager):
    """
        Class to handle text similarity measures.
    """

    def __init__(self):
        """
            Initialize the ImageSimilarityService.
        """
        super().__init__(['SIMILARITY_PERCENTAGE', 'ENABLE_PREPROCESS_TEXT'])
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logger_level)
        self.logger.info('Initializing Image Similarity service...')
        self.similarity_percentage = float(self.env_vars['SIMILARITY_PERCENTAGE'])
        self.enable_preprocess_text = self.env_vars['ENABLE_PREPROCESS_TEXT'].lower() == "true"

    def preprocess_text(self, text):
        """
            Preprocess and tokenize text.

            Parameters:
                text (str): Raw text string.

            Returns:
                str: Preprocessed and tokenized text.
        """
        if not self.enable_preprocess_text:
            return text

        # Retain only letters, numbers, and spaces
        text = re.sub(r'[^a-z0-9 ]', '', text)

        # Replacement rules for uppercase
        replace_rules_upper = str.maketrans('TDCLUEZOBSY', '70GIVF20857')

        # Replacement rules for lowercase
        replace_rules_lower = str.maketrans('ucibogqzsy', 've16099257')

        # Convert to uppercase and apply the first replacement rule
        text = text.upper().translate(replace_rules_upper)

        # Convert to lowercase and apply the second replacement rule
        text = text.lower().translate(replace_rules_lower)

        self.logger.debug("Text preprocessing completed")
        return text

    @staticmethod
    def calculate_cosine_similarity(vec1, vec2):
        """
            Calculate cosine similarity between two vectors.

            Parameters:
                vec1 (array-like): First vector.
                vec2 (array-like): Second vector.

            Returns:
                float: Cosine similarity value.
        """
        return cosine_similarity(vec1, vec2)[0][0]

    def calculate_bow_similarity(self, text1, text2):
        """
            Calculate Bag-of-Words (BoW) similarity.

            Parameters:
                text1 (str): First text string.
                text2 (str): Second text string.

            Returns:
                float: BoW similarity value.
        """
        if not text1 or not text2:
            return 0
        try:
            vectorizer = CountVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            return self.calculate_cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
        except Exception as e:
            self.logger.exception("Exception occurred in calculate_bow_similarity", exc_info=e)
            return 0

    def calculate_tfidf_similarity(self, text1, text2):
        """
            Calculate TF-IDF similarity.

            Parameters:
                text1 (str): First text string.
                text2 (str): Second text string.

            Returns:
                float: TF-IDF similarity value.
        """
        if not text1 or not text2:
            return 0
        try:
            vectorizer = TfidfVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            return self.calculate_cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
        except Exception as e:
            self.logger.exception("Exception occurred in calculate_tfidf_similarity", exc_info=e)
            return 0

    def compare_texts(self, target, text_to_compare):
        """
            Compare two text strings using both BoW and TF-IDF.

            Parameters:
                target (str): Target text string.
                text_to_compare (str): Text string to compare against target.

            Returns:
                float: Averaged similarity value.
        """
        if not target or not text_to_compare:
            self.logger.warning("One or both texts are empty, skipping comparison")
            return 0

        bow_similarity = self.calculate_bow_similarity(target, text_to_compare)
        tfidf_similarity = self.calculate_tfidf_similarity(target, text_to_compare)

        self.logger.debug(f"Similarity scores calculated: BOW={bow_similarity}, TFIDF={tfidf_similarity}")
        return (bow_similarity + tfidf_similarity) / 2 * 100

    def is_similar(self, target_text, text_to_compare):
        """
            Determine if two texts are similar based on a predefined threshold.

            Parameters:
                target_text (str): Target text string.
                text_to_compare (str): Text string to compare.

            Returns:
                tuple: (True if similar, False otherwise, similarity value).
        """
        result = self.compare_texts(target_text, text_to_compare)
        self.logger.debug(f"Calculated similarity score: {result}, Threshold: {self.similarity_percentage}")
        return (True, result) if result >= self.similarity_percentage else (False, result)
