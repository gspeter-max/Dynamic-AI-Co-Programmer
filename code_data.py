import os
import requests
import io
import zipfile
# from dotenv import load_dotenv
from google.colab import userdata

dynamic_api = userdata.get('dynamic_api')
# load_dotenv('/workspaces/Dynamic-AI-Co-Programmer/dynamic_api.env')
account_name = 'scikit-learn'
repo_name = 'scikit-learn'
target_file_name = f'/content/drive/MyDrive/downloaded/{account_name}'
hitting_url = f'https://api.github.com/repos/{account_name}/{repo_name}/zipball/main'
# api_key = os.getenv('dynamic_api')                                              # github api key ( token )


headers = {
       'Accept': 'application/vnd.github.v3+.json'                              # that is customize mime type from github so for that resone we are write
                                                                                # vnd --> vender ( github do that for supporting diff diff versions
                                                                                # git not use boring public mime type
        }

if dynamic_api:
    headers['Authorization'] = f'token {dynamic_api}'                               # you know headers define the http requests

response = requests.get(url = hitting_url ,
        headers = headers ,
        stream = True,                                                            # 1. 'stream'  for when you say to github keep the connection
                                                                                # 2. and first give me some basic information about that like ( sorce code etc )
                                                                                # 3. when i am call ( response.content or response.iter_content ) then return me
                                                                                # whole thing( entire code )
        allow_redirects = True

                                                                                # 'all_redirects = True' mean some thing that url that to you hit that say you go to next url
                                                                                # and nexturl say go to next like ( https://..example1 --> https://...example2)
                                                                                # using that that automatically handle that and reach that point that the exact data is available
                                                                                #( because github create temporary url for geting data
)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z :                        # Bytes --> convert raw data for file_like object ' becuase zipfile.ZipFile only aspect file_object
                                                                                # and inside that response.content have rew data about that
    n_repos = list(set(repo.split('/')[0] for repo in z.namelist()))
    if len(n_repos) == 1:
        repo_name = n_repos[0] + '/'

        for repo_inside_file in z.infolist():
            if not repo_inside_file.is_dir():
                file_name = repo_inside_file.filename[len(repo_name):]
                if file_name[-2:] == 'py':

                    print(file_name)
                    full_name = os.path.join(target_file_name,file_name)

                    os.makedirs(os.path.dirname(full_name),exist_ok = True )

                    with open(full_name, 'wb') as f:
                        f.write(z.read(repo_inside_file.filename))
