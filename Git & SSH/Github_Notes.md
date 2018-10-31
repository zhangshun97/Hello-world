# Git & Github

---

## 第一次配置Github

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

   - `-t` 指定密钥类型，默认是 rsa ，可以省略

   - `-C` 设置注释文字，比如邮箱

   - `-f` 指定密钥文件存储文件名

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

---

## 进一步使用和了解 Git

1. Git 是一个分布式版本控制系统（DVCS），不会因为某一个server的崩溃导致数据/文件的丢失
2. 你可以随时随地使用 Git，因为 Git 对于版本控制的操作大多都可以在本地完成，只需要偶尔接入 Internet 上传（push）一下就可以了
3. **Git 的三个重要状态**：
   1. 已提交（**Commited**）意味着数据已经安全地储存在本地数据库内
   2. 已修改（**Modified**）意味着已经对文件作了改动但是还没有传入数据库内
   3. 已暂存（**Staged**）意味着你标记了一个被改动过的文件当前版本，并且将会在下一次`commit`时一并提交该文件的当前版本

### 关于提交

1. 对于 Git 来说，文件可以分为 已跟踪/未跟踪 两类，以下是 Git 文件的状态变化周期

   ![Git file life-cycle](assets/lifecycle.png)

   1. `git status` 查看文件的 跟踪状态

   2. `git add file_name` 开始跟踪某个文件

      这是个多功能命令：可以用它开始跟踪新文件，或者把已跟踪的文件放到暂存区，还能用于合并时把有冲突的文件标记为已解决状态等

   3. `git status -s` 查看文件状态的**精简信息**，例如

      ```conso
      $ git status -s
       M README
      MM Rakefile
      A  lib/git.rb
      M  lib/simplegit.rb
      ?? LICENSE.txt
      ```

      新添加的未跟踪文件前面有 `??` 标记，新添加到暂存区中的文件前面有 `A` 标记，修改过的文件前面有 `M` 标记。 你可能注意到了 `M` 有两个可以出现的位置，出现在右边的 `M` 表示该文件被修改了但是还没放入暂存区，出现在靠左边的 `M` 表示该文件被修改了并放入了暂存区。 例如，上面的状态报告显示： `README` 文件在工作区被修改了但是还没有将修改后的文件放入暂存区,`lib/simplegit.rb` 文件被修改了并将修改后的文件放入了暂存区。 而 `Rakefile` 在工作区被修改并提交到暂存区后又在工作区中被修改了，所以在暂存区和工作区都有该文件被修改了的记录。

2. 添加 `.gitignore` 忽略文件

   文件 `.gitignore` 的格式规范如下：

   - 所有空行或者以 `＃` 开头的行都会被 Git 忽略。
   - 可以使用标准的 glob 模式匹配。
   - 匹配模式可以以（`/`）开头防止递归。
   - 匹配模式可以以（`/`）结尾指定目录。
   - 要忽略指定模式以外的文件或目录，可以在模式前加上惊叹号（`!`）取反。

   所谓的 glob 模式是指 shell 所使用的简化了的**正则表达式**。 星号（`*`）匹配零个或多个任意字符；`[abc]`匹配任何一个列在方括号中的字符（这个例子要么匹配一个 a，要么匹配一个 b，要么匹配一个 c）；问号（`?`）只匹配一个任意字符；如果在方括号中使用短划线分隔两个字符，表示所有在这两个字符范围内的都可以匹配（比如 `[0-9]` 表示匹配所有 0 到 9 的数字）。 使用两个星号（`*`) 表示匹配任意中间目录，比如`a/**/z` 可以匹配 `a/z`, `a/b/z` 或 `a/b/c/z`等。

3. 使用 `git diff` 来显示尚未暂存（**not staged**）的改动，使用 `git diff --staged` 来显示已暂存的改动

4. 使用 `git commit` 来进行细致化编写提交注释，或者使用 `git commit -m "your_message"` 来进行简单提交说明

5. **一键提交，跳过暂存**，`git commit -a -m "your_message"` 可以跳过 `git add` 步骤直接提交所有**已跟踪**的文件（如果是untracked文件则没有被提交）

### 关于移除/移动

1. `git rm`
2. `git mv`

### 查看提交历史

1. `git log`, more details [here](https://git-scm.com/book/zh/v2/Git-%E5%9F%BA%E7%A1%80-%E6%9F%A5%E7%9C%8B%E6%8F%90%E4%BA%A4%E5%8E%86%E5%8F%B2)

### 关于撤销

1. 重置提交，`git commit --amend`
2. 撤销暂存，use `git reset HEAD <file>...` to unstage
3. 撤销改动，use `git checkout -- <file>...` to discard changes in working directory

### 关于远程仓库

1. `git remote`, more details [here](https://git-scm.com/book/zh/v2/Git-%E5%9F%BA%E7%A1%80-%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E7%9A%84%E4%BD%BF%E7%94%A8)



---

## Reference

- [Git - Book](http://git-scm.com/book)