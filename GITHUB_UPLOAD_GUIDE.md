# Step-by-Step Guide: Upload to GitHub

## Step 1: Create GitHub Repository

1. Go to [https://github.com](https://github.com)
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in:
   - **Repository name**: `chatbot-text-to-sql` (or any name you like)
   - **Description**: "Natural language to SQL chatbot using Streamlit, LangChain, and Ollama"
   - **Visibility**: Choose Public or Private
   - ⚠️ **DO NOT** check "Add a README file" (you already have one)
   - ⚠️ **DO NOT** check "Add .gitignore" (you already have one)
   - ⚠️ **DO NOT** add a license yet
5. Click **"Create repository"**

## Step 2: Open Terminal/Command Prompt

- Press `Windows Key + R`
- Type `cmd` or `powershell` and press Enter
- Navigate to your project folder:
  ```bash
  cd "C:\Users\admin\Documents\CHATBOT TEXT TO SQL 1"
  ```

## Step 3: Initialize Git

```bash
git init
```

## Step 4: Add All Files

```bash
git add .
```

## Step 5: Make Your First Commit

```bash
git commit -m "Initial commit: Chatbot Text to SQL application"
```

## Step 6: Rename Branch to Main

```bash
git branch -M main
```

## Step 7: Connect to GitHub

Replace `chatbot-text-to-sql` with your actual repository name if different:

```bash
git remote add origin https://github.com/niainsyrh/chatbot-text-to-sql.git
```

## Step 8: Push to GitHub

```bash
git push -u origin main
```

You will be asked for your GitHub username and password/token:
- **Username**: `niainsyrh`
- **Password**: Use a Personal Access Token (not your regular password)

### If you need a Personal Access Token:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "Chatbot Project"
4. Select scopes: Check `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing

## Step 9: Verify Upload

1. Go to `https://github.com/niainsyrh/chatbot-text-to-sql` (or your repo name)
2. You should see all your files there!

---

## Troubleshooting

### If "git" command not found:
- Download Git from: [https://git-scm.com/download/win](https://git-scm.com/download/win)
- Install it and restart your terminal

### If authentication fails:
- Use Personal Access Token instead of password
- Make sure you copied the entire token

### If you get "remote origin already exists":
```bash
git remote remove origin
git remote add origin https://github.com/niainsyrh/chatbot-text-to-sql.git
```

### If push is rejected:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

