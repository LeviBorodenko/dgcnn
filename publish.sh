git add .
git commit -m "$1"
git add .
git commit -m "$1"
git push

git checkout gh-pages
python setup.py build
git add .
git commit -n -m $1
git push
git checkout master
