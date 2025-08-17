# Onboarding lectures

## Linux

/bit _-[executables]-_  
/home _-[хомяк]-_  
/var/log

> \$ sudo su _-[зайти в под root]-_

When first time ssh on server, do:

> \$ ssh-copy-id -i ~/.ssh/id*ysda.pub root@147.45.153.149 *-[here .pub; use -f to rewrite server's authorized_keys]-\_

Type password, it will create id_ysda in servers .ssh, and then it won't ask you to do it. Then check that server has ~/.ssh/authorized_keys [with rw permissions] with your .pub

> \$ ssh -i ~/.ssh/id*ysda root@147.45.153.149 *-[here private key]-\_

Create ~/.ssh/config , ask GPT how to work with it!

### Secure copy

> \$ scp -r/path/to/local/file username@remote_host:/path/to/remote/destination  
> \$ scp username@remote_host:/path/to/remote/file /path/to/local/destination

Use _-c blowfish_ for speading the process

### linux permissions

chmod 600 text.txt

\_ \_ \_ - user, group, everyone  
[read write execute] == [4 2 1];

never user 777, 600 is common.

### tmux

tmux server -> many sessions -> many windows -> many panes

> \$ tmux  
> \$ tmux new -s session*name  
> Ctrl + A, then d *-[to exit this window]-\_  
> \$ tmux ls  
> \$ tmux attach -t session_name

Create config for tmux hotkeys.

> \$ tmux source-file ~/.tmux.conf

> \$ tmux kill-session -t session*name  
> \$ tmux kill-server *-[to kill all sessions]-\_

If you are in nested tmux: use Ctrl + A many times

Try https://github.com/tmux-python/libtmux !!!

### Cron (crontab)

Used to schedule tasks (called “cron jobs”) to run automatically at specific intervals (e.g., daily, weekly, monthly).

### Other useful

> \$ ps -aux  
> \$ btop _-[nmon, glance, htop, atop]-_ > \$ kill -9 \_PID\_

## Git

### clone

> \$ git clone --depth=150 ... _-[if lots of commits in project]-_ > \$ git log --oneline --graph  
> \$ git diff _-[local vs added]-_  
> \$ git diff --staged _-[added vs commited]-_

> \$ cat .git/HEAD

> \$ git branch  
> \$ git branch -a

> \$ git branch new_branch_name  
> \$ git checkout new_branch_name

Functionality of **checkout** == functionality of **switch** & **restore**.

### merge vs rebase

> \$ git checkout master

> \$ git rebase new*branch_name *-[moves the entire branch (commits) to the tip of another branch, effectively rewriting history]-_  
> \$ git merge new_branch_name _-[new_branch_name -> master; if there are no conflicts -> **fast-forward**]-\_

**git rebase** : when you want a clean, linear history; used mainly for local branches.  
**git merge** : when you want to preserve the full history; used mainly for collaborative branches.

### revert vs reset

> \$ git revert commit*hash  
> \$ git reset commit_hash *-[reset the state of your project by moving the HEAD]-\_

Reset options:  
--soft: Moves the HEAD pointer only, leaving the staging area and working directory untouched.  
--mixed (default): Moves the HEAD pointer and resets the staging area, but leaves the working directory unchanged.  
--hard: Moves the HEAD pointer, resets the staging area, and updates the working directory, discarding any uncommitted changes.

**git reset** : rewrite history and alter the commit tree; used mainly for local branches.  
**git revert** : creates a new commit that reverses the changes introduced by the specified commit without rewriting the commit history; used mainly for collaborative branches.

### commit --ammend

don't do

> \$ git commit -m "oops forgot to add to previous commit"

do

> \$ git commit --amend _-[and you will refactor previous commit with current added]-_

### stash

To keep working process if you want to switch to another branch without commiting:

```sh
git stash
git stash apply            // reapply stashed changes, but keeps it in the stash list
git stash pop              // with removing stash
git stash clear
```

### clean

> \$ git clean --dry-run _-[to see what would be removed]-_  
> \$ git clean --force  
> \$ git clean -i

### origin & remote

A bare repository contains only the version control data in .git and none of the actual project files. It is primarily used for central repositories where people push and pull code but do not make direct edits to files.

> \$ git init --bare

Then cd .. and:

> \$ git clone origin local

> \$ git remote -v  
> \$ git remote

**git pull origin** == **git fetch** & **git merge origin/master**.

## Yandex GPU's

### Setup

```bash
ssh-keygen -t ed25519   # enter name for_shadgpu
ssh-copy-id -i ~/.ssh/for_shadgpu.pub balabaevvl@45.93.203.34

vim ~/.ssh/config
```

and paste:

```txt
Host shad-gpu
    HostName 45.93.203.34
    User balabaevvl
    IdentityFile ~/.ssh/for_shadgpu
```

### ...

```bash
quota -vs
du -ch -d 1 ~/

pip install ... --user
```

Install apt library locally:

```bash
apt-get dowload MY_PACKAGE
dpkg -x MY_PACKAGE_INSTALLER.deb $HOME
vim ~/.profile
```

And there you write:

```txt
if [ -d "\$\{HOME\}/usr/"]
then
    PATH="\$\{HOME\}/usr/share:\$\{HOME\}/usr/games ... :\$\{PATH\}"
fi
```

```bash
source ./profile
```

or restart terminal.

## .sh

> \$ chmod u+x script.sh

in executable file in first row add shebang:

> #! /usr/bin/env bash

or

> #! /usr/bin/env python3

or ...  
it is used for providing PATH of interpreter to executable.

## docker

> \$ docker build -t ysda_jupyter -f Dockerfile_jupyter .  
> \$ docker run -it -p 20000:9999 ysda_jupyter

and then go to localhost:20000.

[  
If you're using VPN use --network="host":

> \$ docker build --network="host" -t ysda_jupyter -f Dockerfile_jupyter .  
> \$ docker run --network="host" -it ysda_jupyter

and then go to localhost:9999.  
]

To mount directories from the host to the container [:ro to read-only]:

> \$ docker run --network="host" -v ${PWD}/dir1:/home/vbalab/dir1 -v ${PWD}/dir2:/home/vbalab/dir2:ro -it ysda_jupyter

to check all ports (sudo to see programs):

> \$ sudo netstat -nultp

to check docker ports:

> \$ docker ps
