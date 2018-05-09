## Github

1. 首先需要创建global name 和 email ，这是用于信息标记

   `$ git config —global user.name 'name'`

   `$ git config —global user.email 'email address'`

2. 其次需要创建属于自己的SSH Keys

   创建之前可以先检查一下是否已经有keys存在 

   `$ cd ~/.ssh`

   `$ ls`

   如果没有此目录，则可以进行创建，如果有也可以通过再次创建而覆盖

   `$ ssh-keygen -t rsa -C "your_email@example.com"`

   代码参数含义：

   -t 指定密钥类型，默认是 rsa ，可以省略

   -C 设置注释文字，比如邮箱

   -f 指定密钥文件存储文件名

3. 将本地的rsa.pub文件中的公钥密码update到你的GitHub账号中

   在Github个人的Settings里面建立SSH-Keys

4. 在本地的终端测试是否已经匹配完成

   `$ ssh -T git@github.com`

5. 接下来比较麻烦的一步是如何让自己在每一次`git push`的时候不需要输入用户名和密码：

   Plan A（HTTPS）：在终端输入`git config --global credential.helper store`

   Plan B（SSH）：将链接url配置成SSH

   ​	`git remote rm origin`

   ​	`git remote add origin git@github.com:name/code.git`

   ​	`git push origin`

