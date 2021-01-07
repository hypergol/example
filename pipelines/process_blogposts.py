import os
import fire
from hypergol import HypergolProject
from hypergol import Pipeline
from tasks.load_html_pages_task import LoadHtmlPagesTask
from tasks.create_article_texts_task import CreateArticleTextsTask
from tasks.create_articles_task import CreateArticlesTask
from tasks.create_sentences_task import CreateSentencesTask
from data_models.article import Article
from data_models.article_text import ArticleText
from data_models.article_page import ArticlePage
from data_models.sentence import Sentence


def process_blogposts(threads=1, force=False):
    project = HypergolProject(dataDirectory=f'{os.environ["BASE_DIR"]}/tempdata', force=force)
    SOURCE_PATTERN = f'{os.environ["BASE_DIR"]}/data/blogposts/pages_*.pkl'
    articles = project.datasetFactory.get(dataType=Article, name='articles')
    articleTexts = project.datasetFactory.get(dataType=ArticleText, name='article_texts')
    articlePages = project.datasetFactory.get(dataType=ArticlePage, name='article_pages')
    sentences = project.datasetFactory.get(dataType=Sentence, name='sentences')
    loadHtmlPagesTask = LoadHtmlPagesTask(
        outputDataset=articlePages,
        sourcePattern=SOURCE_PATTERN
    )

    createArticleTextsTask = CreateArticleTextsTask(
        inputDatasets=[articlePages],
        outputDataset=articleTexts,
    )

    createArticlesTask = CreateArticlesTask(
        inputDatasets=[articleTexts],
        outputDataset=articles,
        spacyModelName='en_core_web_sm',
        threads=2
    )

    createSentencesTask = CreateSentencesTask(
        inputDatasets=[articles],
        outputDataset=sentences,
    )

    pipeline = Pipeline(
        tasks=[
            loadHtmlPagesTask,
            createArticleTextsTask,
            createArticlesTask,
            createSentencesTask,
        ]
    )
    pipeline.run(threads=threads)


if __name__ == '__main__':
    fire.Fire(process_blogposts)
