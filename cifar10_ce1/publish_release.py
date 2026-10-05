"""Maintainer command: push the committed CE1 branch and publish verified release assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def token_for(root):
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token:
        return token.strip()
    if shutil.which('gh'):
        result = subprocess.run(['gh','auth','token'],capture_output=True,text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    result = subprocess.run(['git','credential','fill'],cwd=root,capture_output=True,text=True,
        input='protocol=https\nhost=github.com\n\n',env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
    if result.returncode == 0:
        values = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if values.get('password'):
            return values['password']
    raise RuntimeError('No GitHub credentials. Configure gh auth login, a Git credential helper, or GITHUB_TOKEN locally.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'cifar10_ce1/weights.json').read_text())
    repo,tag = manifest['repository'],manifest['tag']
    branch = 'experiment/cifar10-ce1'
    def git(*flags,env=None):
        return subprocess.check_output(['git',*flags],cwd=root,env=env,text=True).strip()
    if git('branch','--show-current') != branch or git('status','--porcelain'):
        raise RuntimeError('Commit the complete release on '+branch+' before publishing')
    for name,asset in manifest['assets'].items():
        path = args.artifacts/name
        if not path.is_file() or path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise RuntimeError('Missing or mismatched release asset: '+name)
    token = token_for(root)
    base = 'https://api.github.com/repos/'+repo
    def request(method,url,payload=None,upload=None,allow404=False):
        headers = {'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                   'X-GitHub-Api-Version':'2022-11-28','User-Agent':'cifar10-ce1-release'}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers['Content-Type'] = 'application/json'
        if upload:
            headers['Content-Type'] = 'application/octet-stream'
            headers['Content-Length'] = str(upload.stat().st_size)
            body = upload.open('rb')
        try:
            req = urllib.request.Request(url,data=body,method=method,headers=headers)
            with urllib.request.urlopen(req,timeout=1800) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404 and allow404:
                return None
            raise RuntimeError('GitHub API HTTP %d: %s'%(error.code,error.read().decode()[:500])) from None
        finally:
            if upload:
                body.close()
    info = request('GET',base)
    if not info.get('permissions',{}).get('push'):
        raise RuntimeError('Credential lacks push permission for '+repo)
    head = git('rev-parse','HEAD')
    release = request('GET',base+'/releases/tags/'+tag,allow404=True)
    if release and not release['draft']:
        raise RuntimeError('This release is already public; inspect it before attempting a new publication')
    # Askpass contains no credential; the token is supplied only in the child environment.
    with tempfile.TemporaryDirectory() as tmp:
        askpass = Path(tmp)/'askpass.py'
        askpass.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["CE1_PUBLISH_TOKEN"])\n')
        askpass.chmod(0o700)
        env = dict(os.environ,GIT_ASKPASS=str(askpass),GIT_TERMINAL_PROMPT='0',CE1_PUBLISH_TOKEN=token)
        git('push','https://github.com/'+repo+'.git','HEAD:refs/heads/'+branch,env=env)
    if release is None:
        notes = (root/'cifar10_ce1/RELEASE_NOTES.md').read_text()
        release = request('POST',base+'/releases',dict(tag_name=tag,target_commitish=head,
            name='CIFAR-10 CE1 +50K: one-step classification',body=notes,draft=True,prerelease=False))
    elif release.get('target_commitish') not in (head,branch):
        raise RuntimeError('Existing draft points to a different commit')
    existing = {a['name']:a for a in release['assets']}
    for name,asset in manifest['assets'].items():
        if name in existing:
            uploaded = existing[name]
        else:
            print('Uploading',name,flush=True)
            upload_url = release['upload_url'].split('{')[0]+'?'+urllib.parse.urlencode({'name':name})
            uploaded = request('POST',upload_url,upload=args.artifacts/name)
        expected_digest = 'sha256:'+asset['sha256']
        if uploaded['size'] != asset['bytes'] or uploaded.get('digest') != expected_digest:
            raise RuntimeError('Uploaded asset failed size/SHA256 verification: '+name)
        print('Verified uploaded',name,flush=True)
    sums = args.artifacts/'SHA256SUMS'
    if 'SHA256SUMS' not in existing:
        request('POST',release['upload_url'].split('{')[0]+'?name=SHA256SUMS',upload=sums)
    published = request('PATCH',base+'/releases/'+str(release['id']),dict(draft=False))
    print(json.dumps({'branch':'https://github.com/'+repo+'/tree/'+branch,
                      'commit':head,'release':published['html_url'],
                      'assets':[a['browser_download_url'] for a in published['assets']]},indent=2))


if __name__ == '__main__':
    main()
