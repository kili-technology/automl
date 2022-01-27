import click


@click.command()
@click.option('--api-key', default='', help='Kili API Key')
@click.option('--project-id', default='', help='Kili project ID')
def main(api_key: str, project_id: str):
    print(api_key, project_id)


if __name__ == "__main__":
    main()