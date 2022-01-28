import click

from utils.helpers import kili_print


@click.command()
@click.option('--api-key', default='', help='Kili API Key')
@click.option('--project-id', default='', help='Kili project ID')
def main(api_key: str, project_id: str):
    '''
    '''
    kili_print('not implemented yet')


if __name__ == "__main__":
    main()