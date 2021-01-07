from datetime import datetime
from hypergol import Task
from bs4 import BeautifulSoup
from data_models.article_text import ArticleText


class CreateArticleTextsTask(Task):

    def __init__(self, *args, **kwargs):
        super(CreateArticleTextsTask, self).__init__(*args, **kwargs)


    def run(self, articlePage):
        try:
            soup = BeautifulSoup(articlePage.body, features='html.parser')
            body = soup.find('div', class_='has-content-area')
            content = '\n'.join([v.text for v in body.children])
            meta = soup.find('meta', property="article:published_time")
            self.output.append(ArticleText(
                articleTextId=articlePage.articlePageId,
                publishDate=datetime.fromisoformat(meta.attrs['content']),
                title=body['data-title'],
                text=content,
                url=articlePage.url
            ))
        except AttributeError as ex:
            print(f'Error in {articlePage.articlePageId}: {ex}')
            self.output.append(ArticleText(
                articleTextId=articlePage.articlePageId,
                publishDate=datetime.now(),
                title="Error while processing article",
                text=str(ex),
                url=articlePage.url
            ))

    def finish_job(self, jobReport):
        # TODO: Update jobReport after the last iteration. Close file handlers or release memory of non-python objects here if necessary
        pass

    def finish_task(self, jobReports, threads):
        # User-defined finalisation at the end of the task.
        pass
