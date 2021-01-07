import spacy
from hypergol import Task
from data_models.article import Article
from data_models.sentence import Sentence
from data_models.token import Token


class CreateArticlesTask(Task):

    def __init__(self, spacyModelName, *args, **kwargs):
        super(CreateArticlesTask, self).__init__(*args, **kwargs)
        self.spacyModelName = spacyModelName
        self.spacyModel = None

    def init(self):
        self.spacyModel = spacy.load(self.spacyModelName)

    def run(self, articleText):
        article = Article(
            articleId=articleText.articleTextId,
            url=articleText.url,
            publishDate=articleText.publishDate,
            title=articleText.title,
            text=articleText.text,
            sentences=[]
        )
        for k, sentence in enumerate(self.spacyModel(articleText.text).sents):
            tokenOffset = sentence[0].i
            tokenCharOffset = sentence[0].idx
            article.sentences.append(Sentence(
                startChar=sentence[0].idx,
                endChar=sentence[-1].idx+len(str(sentence[-1])),
                articleId=article.articleId,
                sentenceId=k,
                tokens=[Token(
                    i=token.i-tokenOffset,
                    startChar=token.idx-tokenCharOffset,
                    endChar=token.idx+len(str(token))-tokenCharOffset,
                    depType=token.dep_,
                    depHead=token.head.i-tokenOffset,
                    depLeftEdge=token.left_edge.i-tokenOffset,
                    depRightEdge=token.right_edge.i-tokenOffset,
                    posType=token.pos_,
                    posFineType=token.tag_,
                    lemma=token.lemma_,
                    text=token.text
                ) for token in sentence]
            ))
        self.output.append(article)

    def finish_job(self, jobReport):
        # TODO: Update jobReport after the last iteration. Close file handlers or release memory of non-python objects here if necessary
        pass

    def finish_task(self, jobReports, threads):
        # User-defined finalisation at the end of the task.
        pass
