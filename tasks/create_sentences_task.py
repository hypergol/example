from hypergol import Task
from data_models.article import Article
from data_models.sentence import Sentence


class CreateSentencesTask(Task):

    def __init__(self, *args, **kwargs):
        super(CreateSentencesTask, self).__init__(*args, **kwargs)

    def run(self, article):
        for sentence in article.sentences:
            self.output.append(sentence)

    def finish_job(self, jobReport):
        # TODO: Update jobReport after the last iteration. Close file handlers or release memory of non-python objects here if necessary
        pass

    def finish_task(self, jobReports, threads):
        # User-defined finalisation at the end of the task.
        pass
