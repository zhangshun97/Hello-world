## Command line instructions for Git

### Git global setup
`git config --global user.name "Shun Zhang"`  
`git config --global user.email "zhangs15@fudan.edu.cn"` 

### Create a new repository
`git clone ssh://git@10.190.2.125:20022/zhangshun/asd.git`  
`cd asd`  
`touch README.md`  
`git add README.md`  
`git commit -m "add README"`  
`git push -u origin master`  

### Existing folder
`cd folder_name`  
`git init`  
`git remote add origin ssh://git@10.190.2.125:20022/zhangshun/asd.git`  
`git add .`  
`git commit -m "Initial commit"`  
`git push -u origin master`

### Existing Git repository
`cd existing_repo`  
`git remote rename origin old-origin`  
`git remote add origin ssh://git@10.190.2.125:20022/zhangshun/asd.git`  
`git push -u origin --all`  
`git push -u origin --tags`

