The code is central to calculating 2D landscape metrics.
Data was obtained from the GEE, with the code as follows:

// 加载数据集
var CFATD = ee.ImageCollection("projects/ee-caiyt33-catcd/assets/China_Forest_AGB_TimeSeries");
var CFATD_2023 = CFATD.filter(ee.Filter.eq('year',2023)).first()
Map.centerObject(CFATD_2023, 4)
Map.addLayer(CFATD_2023.select('Uncertainty'),{min:0,max:30,palette:['white','red']},'2023 Uncertainty')
Map.addLayer(CFATD_2023.select('AGB'),{min:0,max:200,palette:['white','green']},'2023 AGB')
// 查看年份范围
var years = CFATD.aggregate_array('year').distinct().sort();
print('Available years:', years);

var yearsList = years.getInfo(); // 同步获取（在 Code Editor 会阻塞），得到普通数组
for (var i = 0; i < yearsList.length; i++) {
  var y = yearsList[i];
  var img = CFATD.filter(ee.Filter.eq('year', y)).first();
  var agb = img.select('AGB');

  Export.image.toDrive({
    image: agb,
    description: 'AGB_' + y,
    folder: 'CFATD_AGB',
    fileNamePrefix: 'AGB_' + y,
    scale: 30,
    region: img.geometry().bounds(),
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}


