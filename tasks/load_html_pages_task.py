import glob
import pickle
from hypergol import Job
from hypergol import Task
from data_models.article_page import ArticlePage


class LoadHtmlPagesTask(Task):

    def __init__(self, sourcePattern, *args, **kwargs):
        super(LoadHtmlPagesTask, self).__init__(*args, **kwargs)
        self.sourcePattern = sourcePattern

    def init(self):
        # TODO: initialise members that are NOT "Delayed" here (e.g. load spacy model)
        pass

    def get_jobs(self):
        filenames = glob.glob(self.sourcePattern)
        return [Job(id_=k, total=len(filenames), parameters={'id': k, 'filename': filename}) for k, filename in enumerate(filenames)]

    def source_iterator(self, parameters):
        jobId = parameters['id']
        pages = pickle.load(open(parameters['filename'], 'rb'))
        for k, page in enumerate(pages):
            page['pageId'] = jobId*100_000 + k
            yield (page, )

    def run(self, data):
        self.output.append(ArticlePage(
            articlePageId=data['pageId'],
            url=data['link'],
            body=data['page']
        ))

    def finish_job(self, jobReport):
        # TODO: Update jobReport after the last iteration. Close file handlers or release memory of non-python objects here if necessary
        pass

    def finish_task(self, jobReports, threads):
        # User-defined finalisation at the end of the task.
        pass
