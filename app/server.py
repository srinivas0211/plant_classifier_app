from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=18cRRo0K1bwL7Oj0f4BmcxWogH3O1-F_g'
export_file_name = 'stage-2-34.pth'

classes = ['African violet','Agave plant','Alocasia','Alocasia Nebula','Aloe Vera','Amaryllis','Angel Wing Begonia','Anthurium','Aralia','Aralia-Balfour','Areca','Asparagus Fern','Azalea','Baby’s teras','Bamboo','Barberton daisy','Beach spider lily','Begonia','Begonia Rex','Belladonna lily','Bird of Paradise','Bird’s Nest fern','Bleeding heart vine','Boston Fern','Bougainvillea','Bromeliad Guzmania','Bromeliad-Aechmea fasciata','Busy lizzie','Cactus','Caladium','Calathea','Calla Lily','Cat palm','Chinese Evergreen','Chinese Evergreen- Amelia','Christmas cactus','Chrysanthemum','Cineraria','Coleus','Coral berry','Corsage orchid','Creeping fig','Croton','Crown of throns','Ctenanthe','Cyclamen','Desert rose','Dieffenbachia','Donkey’s tail','Dracaena Janet Craig','Dracaena Lemon lime','Dracaena Marginata','Dracaena Massangeana','Dracaena Warnekii','Dracaena compacta','Dracaena reflexa','Dracaena reflexa song of india','Easter Lily','Emerald gem plant','English Ivy','Episcia','Eternal flame','Fatsia','Ficus Alii','Ficus tree','Fiddle leaf fig ','Fishtail palm','Fittonia','Flowering maple','Gardenia','Geranium','Goldfish plant','Grape Ivy','Hawaiian Schefflera','Hawaiian Schefflera- Gold Capella','Heartleaf Philodendron','Hibiscus','Hidu Rope','Hoya','Hoya shooting stars','Jade plant','Jasmine','Kaffir lily','Kalanchoe','Kangaroo paw fern','Kentia palm','Kimberley Queen fern','Lipstick plant','Lollipop plant','Luck bamboo','Maidenhair fern','Majesty palm','Marble queen pothos','Mimosa pudica','Miniature rose','Moses in the cradle','Natal mahogany','Norfolk Island pine','Orchid- Cymbidium','Orchid- Phalaenopsis','Pachira acquatica','Parlor palm','Peace lily','Pencil cactus','Peperomia','Peperomia- caperata','Philodendron Selloum','Philodendron Xanadu','Philodendron congo','Philodendron imperial red','Podocarpus','Poinsettia','Polka dot','Ponytail palm','Pothos','Prayer','Pygmy date','Rabbit’s foot fern','Rhapis','Rubber tree','Sansevieria','Schefflera','Selaginella','Split leaf philodendron','Staghorn fern','Strawberry begonia','Stromanthe tricolor','Swedish Ivy','Wandering Jew','Yucca','Zamioculcas zamiifolia']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
