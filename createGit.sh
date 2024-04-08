echo "# BAM2" >> README.md
rm -rf ./.git
git init
git add README.md
git add .
git check-ignore -v -- *
echo "If eveyrthing ok - please <RETURN> else press Ctrl-C to stop"
read x
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/anushkamanoj16/BAM2.git
git push -u origin main
