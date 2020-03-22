all :
	python src/refresh_data.py;     # get the latest data
	python src/refresh_model.py;    # rebuild the model
	python app.py                   # run the app

refresh :
	python src/refresh_data.py;     # get the latest data
	python src/refresh_model.py;    # rebuild the model

app :
	python app.py                   # run the app

deploy_heroku :
	heroku container:push web --app ubc-mds-github-search 
	heroku container:release web --app ubc-mds-github-search

deploy_heroku_refresh :
	python src/refresh_data.py;     # get the latest data
	python src/refresh_model.py;    # rebuild the model
	heroku container:push web --app ubc-mds-github-search 
	heroku container:release web --app ubc-mds-github-search

clean :
	rm -f data/*.csv
	rm -f data/*.npz
	rm -f data/*.pkl