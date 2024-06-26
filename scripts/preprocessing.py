# import all necessary libraries
from deepmultilingualpunctuation import PunctuationModel
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize
import re
import pandas as pd

# download the punkt data for nltk sent_tokenize
nltk.download('punkt')

class ProcessingPipeline:
    def __init__(self):
        # intialise the variables needed for the initialisation
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bullet_point_pattern = self.initialise_bullet_point_pattern()
        self.contractions = self.initialise_contractions()
        self.initialise_punctuation_model()
        self.initialise_spell_check_model()

    def initialise_bullet_point_pattern(self):
        # define the regex patterns to identify common bullet points
        patterns = [
            r'^\s*[\*\-\•]\s+',       # bullet points (e.g., •, *, -)
            r'^\s*\d+\.\s+',       # number list (e.g., 1., 2., 3.)
            r'^\s*\d+\)\s+',       # number list (e.g., 1), 2), 3))
            r'^\s*[a-zA-Z]\.\s+',  # letter list (e.g., a., b., c.)
            r'^\s*[a-zA-Z]\)\s+',   # letter list (e.g., a), b), c))
            r'^\s*=-\s+' # in the scenario the point form was incorrectly inputted in as =- which occured in the example identified in the text on the 1st line
        ]
        # combine the patterns into a single regex for the bullet points
        bullet_point_pattern = re.compile('|'.join(patterns), re.MULTILINE)
        return bullet_point_pattern
    
    def initialise_contractions(self):
        # set up the regex patterns to identify the existing contractions that may exist
        simple_contractions = {
            re.compile(r"\bcan'?t\b", re.I | re.U): "cannot",
            re.compile(r"\bcan'?t'?ve\b", re.I | re.U): "cannot have",
            re.compile(r"\b'?cause\b", re.I | re.U): "because",
            re.compile(r"\bcould'?ve\b", re.I | re.U): "could have",
            re.compile(r"\bcouldn'?t\b", re.I | re.U): "could not",
            re.compile(r"\bcouldn'?t'?ve\b", re.I | re.U): "could not have",
            re.compile(r"\bdidn'?t\b", re.I | re.U): "did not",
            re.compile(r"\bdoesn'?t\b", re.I | re.U): "does not",
            re.compile(r"\bdon'?t\b", re.I | re.U): "do not",
            re.compile(r"\bdoin'?\b", re.I | re.U): "doing",
            re.compile(r"\bdunno\b", re.I | re.U): "do not know",
            re.compile(r"\bgimme'?\b", re.I | re.U): "give me",
            re.compile(r"\bgoin'?\b", re.I | re.U): "going",
            re.compile(r"\bgonna'?\b", re.I | re.U): "going to",
            re.compile(r"\bhadn'?t\b", re.I | re.U): "had not",
            re.compile(r"\bhadn'?t'?ve\b", re.I | re.U): "had not have",
            re.compile(r"\bhasn'?t\b", re.I | re.U): "has not",
            re.compile(r"\bhaven'?t\b", re.I | re.U): "have not",
            re.compile(r"\bhe'?d'?ve\b", re.I | re.U): "he would have",
            re.compile(r"\bhow'?d\b", re.I | re.U): "how did",
            re.compile(r"\bhow'?d'?y\b", re.I | re.U): "how do you",
            re.compile(r"\bhow'?ll\b", re.I | re.U): "how will",
            re.compile(r"\bI'?d'?ve\b", re.I | re.U): "I would have",
            re.compile(r"\bI'?m\b", re.I | re.U): "I am",
            re.compile(r"\bI'?ve\b", re.I | re.U): "I have",
            re.compile(r"\bisn'?t\b", re.I | re.U): "is not",
            re.compile(r"\bit'?d'?ve\b", re.I | re.U): "it would have",
            re.compile(r"\bkinda\b", re.I | re.U): "kind of",
            re.compile(r"\blet'?s\b", re.I | re.U): "let us",
            re.compile(r"\bma'?am\b", re.I | re.U): "madam",
            re.compile(r"\bmayn'?t\b", re.I | re.U): "may not",
            re.compile(r"\bmight'?ve\b", re.I | re.U): "might have",
            re.compile(r"\bmightn'?t\b", re.I | re.U): "might not",
            re.compile(r"\bmightn'?t'?ve\b", re.I | re.U): "might not have",
            re.compile(r"\bmust'?ve\b", re.I | re.U): "must have",
            re.compile(r"\bmustn'?t\b", re.I | re.U): "must not",
            re.compile(r"\bmustn'?t'?ve\b", re.I | re.U): "must not have",
            re.compile(r"\bnothin'?\b", re.I | re.U): "nothing",
            re.compile(r"\bneedn'?t\b", re.I | re.U): "need not",
            re.compile(r"\bneedn'?t'?ve\b", re.I | re.U): "need not have",
            re.compile(r"\bo'?clock\b", re.I | re.U): "of the clock",
            re.compile(r"\boughta\b", re.I | re.U): "ought to",
            re.compile(r"\boughtn'?t\b", re.I | re.U): "ought not",
            re.compile(r"\boughtn'?t'?ve\b", re.I | re.U): "ought not have",
            re.compile(r"\bshan'?t\b", re.I | re.U): "shall not",
            re.compile(r"\bsha'?n'?t\b", re.I | re.U): "shall not",
            re.compile(r"\bshan'?t'?ve\b", re.I | re.U): "shall not have",
            re.compile(r"\bshe'?d'?ve\b", re.I | re.U): "she would have",
            re.compile(r"\bshould'?ve\b", re.I | re.U): "should have",
            re.compile(r"\bshouldn'?t\b", re.I | re.U): "should not",
            re.compile(r"\bshouldn'?t'?ve\b", re.I | re.U): "should not have",
            re.compile(r"\bso'?ve\b", re.I | re.U): "so have",
            re.compile(r"\bsomethin'?\b", re.I | re.U): "something",
            re.compile(r"\bthat'?d'?ve\b", re.I | re.U): "that would have",
            re.compile(r"\bthere'?d'?ve\b", re.I | re.U): "there would have",
            re.compile(r"\bthey'?d'?ve\b", re.I | re.U): "they would have",
            re.compile(r"\bthey'?re\b", re.I | re.U): "they are",
            re.compile(r"\bthey'?ve\b", re.I | re.U): "they have",
            re.compile(r"\b'?tis\b", re.I | re.U): "it is",
            re.compile(r"\bto'?ve\b", re.I | re.U): "to have",
            re.compile(r"\bu\b(?!\.)", re.I | re.U): "you",
            re.compile(r"\bwasn'?t\b", re.I | re.U): "was not",
            re.compile(r"\bwanna'?\b", re.I | re.U): "want to",
            re.compile(r"\bwe'?d'?ve\b", re.I | re.U): "we would have",
            re.compile(r"\bwe'll\b", re.I | re.U): "we will",
            re.compile(r"\bwe'?ll'?ve\b", re.I | re.U): "we will have",
            re.compile(r"\bwe're\b", re.I | re.U): "we are",
            re.compile(r"\bwe'?ve\b", re.I | re.U): "we have",
            re.compile(r"\bweren'?t\b", re.I | re.U): "were not",
            re.compile(r"\bwhat'?re\b", re.I | re.U): "what are",
            re.compile(r"\bwhat'?ve\b", re.I | re.U): "what have",
            re.compile(r"\bwhen'?ve\b", re.I | re.U): "when have",
            re.compile(r"\bwhere'?d\b", re.I | re.U): "where did",
            re.compile(r"\bwhere'?ve\b", re.I | re.U): "where have",
            re.compile(r"\bwho'?ve\b", re.I | re.U): "who have",
            re.compile(r"\bwhy'?ve\b", re.I | re.U): "why have",
            re.compile(r"\bwill'?ve\b", re.I | re.U): "will have",
            re.compile(r"\bwon'?t\b", re.I | re.U): "will not",
            re.compile(r"\bwon'?t'?ve\b", re.I | re.U): "will not have",
            re.compile(r"\bwould'?ve\b", re.I | re.U): "would have",
            re.compile(r"\bwouldn'?t\b", re.I | re.U): "would not",
            re.compile(r"\bwouldn'?t'?ve\b", re.I | re.U): "would not have",
            re.compile(r"\by'?all\b", re.I | re.U): "you all",
            re.compile(r"\by'?all'?d\b", re.I | re.U): "you all would",
            re.compile(r"\by'?all'?d'?ve\b", re.I | re.U): "you all would have",
            re.compile(r"\by'?all'?re\b", re.I | re.U): "you all are",
            re.compile(r"\by'?all'?ve\b", re.I | re.U): "you all have",
            re.compile(r"\byou'?d'?ve\b", re.I | re.U): "you would have",
            re.compile(r"\byou'?re\b", re.I | re.U): "you are",
            re.compile(r"\byou'?ve\b", re.I | re.U): "you have"
        }
        return simple_contractions
    
    def initialise_punctuation_model(self):
        # initialise the punctuation correction model 
        self.punctuation_model = PunctuationModel()
        
    def initialise_spell_check_model(self):
        # define the path to the spell-check model in huggingface
        path_to_model = "ai-forever/T5-large-spell"

        # initialise the model and tokenizer to use from huggingface
        self.spell_check_tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        self.spell_check_model = T5ForConditionalGeneration.from_pretrained(path_to_model)

        # move the model to the gpu device if available, otherwise the model will use the cpu for computation
        self.spell_check_model.to(self.device)

        # the prefix was defined to specify that the task for the model is for grammar correction
        self.spell_check_prefix = "grammar: "

    def drop_duplicate_columns(self, df, duplicate_columns):
        '''
            Drops the duplicate columns specified in the arguments
        '''
        new_df = df.drop(columns=duplicate_columns) # ['Summary Notes_Summary Notes.1', 'Diagnosis Code.1']
        return new_df
    
    def format_headers(self, df):
        '''
            Replace the spaces in the headers with underscores and convert the headers to lowercase
        '''
        new_df = df.copy()
        new_df.columns = df.columns.str.replace(' ', '_').str.lower()
        return new_df
    
    def feature_selection(self, df, features):
        '''
            Select only the specified columns needed for the downstream tasks. Must include the initials of the PWD.
        '''
        new_df = df[features]
        return new_df

    def format_dataframe(self, df):
        '''
            Formats the dataframe to follow the schema: [pwd, headline, text_collection]

            Parameters:
            df (pd.DataFrame): The input DataFrame containing text data.

            Returns:
            formatted_df (pd.DataFrame): A new DataFrame following the schema of [pwd, headline, text_collection]. 
            This schema makes it easier to process each sentence while preserving contextual clues from the 
            original headers and allows for identifying the PWD associated with the information.
        '''

        # initialize new_rows for the new DataFrame with the schema of ['pwd', 'headline', 'text_collection']
        new_rows = []

        # iterate through each row
        for index, row in df.iterrows():
            # Identify the PWD's name or initial
            pwd = row['about_me_name']

            # Create a copy of the row without the 'about_me_name' column
            row_data = row.drop('about_me_name')

            # Iterate through the text columns and their values within the row
            for column, value in row_data.items():
                # Ensure that empty values are not present in the outputted DataFrame
                if isinstance(value, str) and value:              
                    new_row = [pwd, str(column), str(value)]
                    new_rows.append(new_row)
                    
        # create the new_df using the new rows
        formatted_df = pd.DataFrame(new_rows, columns=['pwd', 'headline', 'text'])
        return formatted_df
    
    def format_text_to_list(self, df):
        '''
        Formats the text in the text_collection column of the formatted DataFrame to be a single element in a list.

        This function ensures that downstream functions will receive input lists
        throughout the pipeline, facilitating modular development and allowing functions 
        to be dropped as needed.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing text data.

        Returns:
        new_df (pd.DataFrame): A new DataFrame where the text columns (excluding about_me_name) are converted to a list.
        '''
        new_df = df.copy()

        # convert the text into list datatype
        new_df['text_collection'] = df['text_collection'].apply(lambda x: [x])
        return new_df

    def clean_outer_spacings(self, text):
        '''
            Cleans the input text of whitespaces that prepend and append to the ends of the text.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text which was processed and cleaned of the outer whitespaces.
        '''
        
        # remove unnecessary linebreaks and spacing at the beginning and end of the text
        processed_text = re.sub(r'^\s+|\s+$', '', text)

        return processed_text

    def clean_inner_spacings(self, text):
        '''
            Cleans the input text of whitespaces that exist in recurring sequences within the text. i.e. hello     world.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text which was processed and cleaned of the inner whitespaces.
        '''

        # remove additional spacing between words within the text but preserve the singular linebreaks i.e \n
        processed_text = re.sub(r'(?<!\n)[ \t]+', ' ', text) # converts any spaces or tab indents to a single space
        processed_text = re.sub(r'(?<!\n)[\n]+', '\n', processed_text) # converts multiple linebreaks which occur in immediate sequence to a single linebreak
        processed_text = re.sub(r'(?<=\n)[ ]+', '', processed_text) # removes spaces that proceed immediately after a linebreak

        return processed_text

    def normalize_casing(self, text):
        '''
            Normalizes the casing of the input text to lowercase.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text which has its casing changed to lowercase.
        '''

        # lowercase the entire text to normalize the casing of the text
        processed_text = text.lower()
        
        return processed_text

    def remove_bullet_points(self, text): # limitation: cannot remove multiple bullet points in a single line
        '''
            Removes the bullet points from the text. This step should be performed before and after the sentences are tokenized.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text which has bullet points removed.
        '''
        
        processed_text = self.bullet_point_pattern.sub('', text)

        return processed_text
    
    def expand_contractions(self, text):
        '''
            Expands the contractions in the text.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text where the contracted words are expanded.
        '''
        # expand the contracted words that exist within the text
        processed_text = self.expand_contractions(text)

        return processed_text
    
    def split_linebreaks(self, df):
        '''
            Splits the text into elementary sentences based solely on the linebreaks that exist within the text.

            Parameters:
            df (pd.DataFrame): The input dataframe to be processed.

            Returns:
            processed_df (pd.DataFrame): The output dataframe where the elementary sentences are exploded to their own rows.
        '''
        # initialise the processed_df by making it a copy of the input df
        processed_df = df.copy()

        # split the text in the text column by their line breaks
        processed_df['text'] = df['text'].apply(lambda x: x.split('\n'))

        # explode the list of elementary sentences into separate rows
        processed_df = processed_df.explode('text')

        # reset index for a cleaner sequential index
        processed_df = processed_df.reset_index(drop=True)
        
        return processed_df
    
    def correct_punctuation(self, text):
        '''
            Corrects the punctuation of the text using the model from hugging face. The model is called oliverguhr/fullstop-punctuation-multilang-large.

            Parameters:
            text (str): The input text to be processed and punctuated.

            Returns:
            corrected_punctuation_text (str): The output text where the text have their punctuations corrected if the punctuation of the text can be corrected.
            OR
            text (str): The original input text will be returned if the punctation of the text cannot be corrected.
        '''
    
        try:
            # correct the punctuation of the text
            corrected_punctuation_text = self.punctuation_model.restore_punctuation(text)
            return corrected_punctuation_text
        except Exception as e:
            # if the sentence is too short to have its punctuation corrected by the model.
            print(f"Too few words to process the following text: {text}. Error: {e}")
            return text
            
    def sentence_tokenization(self, df):
        '''
            Perform sentence tokenization on the text column in the input dataframe.

            Parameters:
            df (pd.DataFrame): The input dataframe to be processed.

            Returns:
            processed_df (pd.DataFrame): The output dataframe where the text are tokenized into elementary sentences and then exploded to their own rows.
        '''

        # initialise the processed_df by making it a copy of the input df
        processed_df = df.copy()

        # tokenize the text into elementary sentences using NLTK
        processed_df['text'] = df['text'].apply(lambda x: sent_tokenize(x))
        
        # explode the list of elementary sentences into separate rows
        processed_df = processed_df.explode('text')

        # reset index for a cleaner sequential index
        processed_df = processed_df.reset_index(drop=True)

        return processed_df
    
    def clean_spelling(self, text):
        '''
            Clean the spelling of the input text.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            corrected_spelling_text (str): The output text which had its spelling corrected.
        '''

        # add the spell_check_prefix to the line specified for spell correction
        sentence = self.spell_check_prefix + text

        # obtain the encodings and decode it to get the corrected sentence
        encodings = self.spell_check_tokenizer(sentence, return_tensors="pt").to(self.device)
        generated_tokens = self.spell_check_model.generate(**encodings, max_new_tokens=512)
        answer = self.spell_check_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return answer
    
    def remove_bullet_points(self, text): # limitation: cannot remove multiple bullet points in a single line
        '''
            Removes the bullet points from the text. This step should be performed after all the sentences were fully tokenized.

            Parameters:
            text (str): The input text to be processed.

            Returns:
            processed_text (str): The output text which has bullet points removed.
        '''
        
        processed_text = self.bullet_point_pattern.sub('', text)

        return processed_text
    
    def perform_preprocessing(self, df, features, duplicate_columns=[], 
                              clean_outer_spacings=True,
                              clean_inner_spacings=True,
                              normalize_casing=True,
                              remove_bullet_points=True,
                              expand_contractions=False,
                              split_linebreaks=True,
                              correct_punctuation=True,
                              sentence_tokenization=True,
                              clean_spelling=True):
        
        processing_df = df.copy()

        # drop the specified duplicate columns
        # processing_df = self.drop_duplicate_columns(processing_df, duplicate_columns)
        
        # format the dataframe headers to be follow a consistent naming scheme
        processing_df = self.format_headers(processing_df)
        self.checkpoint1 = processing_df

        # select the specified features needed in the dataframe
        processing_df = self.feature_selection(processing_df, features)
        self.checkpoint2 = processing_df

        # format the dataframe to have the following schema: [pwd, headline, text_collection]
        processing_df = self.format_dataframe(processing_df)
        self.checkpoint3 = processing_df

        # clean the unnecessary whitespaces that prepend or append to the text
        if clean_outer_spacings==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.clean_outer_spacings(x))
        self.checkpoint4 = processing_df

        # clean the unnecessary whitespaces that exist within the text
        if clean_inner_spacings==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.clean_inner_spacings(x))
        self.checkpoint5 = processing_df

        # normalize the casing of the text in the data entries
        if normalize_casing==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.normalize_casing(x))
        self.checkpoint6 = processing_df
        
        # remove the bullet points from the text
        if remove_bullet_points==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.remove_bullet_points(x))
        self.checkpoint7 = processing_df

        # remove the bullet points from the text
        if remove_bullet_points==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.remove_bullet_points(x))
        self.checkpoint8 = processing_df

        # if expand_contractions==True: processing_df['text_collection'] = processing_df['text_collection'].apply(lambda x: self.expand_contractions(x))
        # self.checkpoint7 = processing_df

        # split the text into sentences based on their linebreaks
        if split_linebreaks==True: processing_df = self.split_linebreaks(processing_df)
        self.checkpoint8 = processing_df

        # correct the punctuation of the text
        if correct_punctuation==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.correct_punctuation(x))
        self.checkpoint9 = processing_df

        # tokenize the sentences
        if sentence_tokenization==True: processing_df = self.sentence_tokenization(processing_df)
        self.checkpoint10 = processing_df

        # remove the bullet points from the text
        if remove_bullet_points==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.remove_bullet_points(x))
        self.checkpoint11 = processing_df
        
        # clean the spelling of the text
        if clean_spelling==True: processing_df['text'] = processing_df['text'].apply(lambda x: self.clean_spelling(processing_df))
        self.processed_df = processing_df

        return self.processed_df