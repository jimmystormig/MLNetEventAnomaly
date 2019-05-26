using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetEventAnomaly.Predictors
{
    public class SsaChangePointPredictor
    {
        private const string ModelPath = "data/SsaChangePointPredictorModel.zip";
        private readonly MLContext _ml;
        private TimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction> _timeSeriesPredictionFunction;
    
        public SsaChangePointPredictor()
        {
            _ml = new MLContext();
        }
    
        public void Initialize()
        {
            try
            {
                using (var file = File.OpenRead(ModelPath))
                    if (file != null && file.Length > 0)
                    {
                        var model = _ml.Model.Load(file, out var schema);
                        _timeSeriesPredictionFunction = model.CreateTimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction>(_ml);
                        return;
                    }
            }
            catch (FileNotFoundException) { }
        
            const int SeasonalitySize = 5;
            const int TrainingSeasons = 5;
            const int TrainingSize = SeasonalitySize * TrainingSeasons;

            var data = File
                .ReadAllLines("training_data/maindoorstates.csv")
                .Select(row => row.Split(';'))
                .Select(row => new { Date = DateTime.Parse(row[0], null, DateTimeStyles.AssumeUniversal), Value = int.Parse(row[1]) })
                .Select(row => new SsaChangePointPredictorData((int) row.Date.DayOfWeek, row.Date.Hour, row.Value))
                .ToList();

            var dataView = _ml.Data.LoadFromEnumerable(data);

            var inputColumnName = nameof(SsaChangePointPredictorData.DoorOpenings);
            var outputColumnName = nameof(SsaChangePointPredictorPrediction.Change);

            _timeSeriesPredictionFunction = _ml.Transforms.DetectChangePointBySsa(outputColumnName, inputColumnName, 95, 8, TrainingSize, SeasonalitySize + 1)
                .Fit(dataView)
                .CreateTimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction>(_ml);
        }

        public SsaChangePointPredictorPrediction Predict(DateTime timestamp, int doorOpenings) => 
            _timeSeriesPredictionFunction.Predict(new SsaChangePointPredictorData((int)timestamp.DayOfWeek, timestamp.Hour, doorOpenings));

        public void SaveModel() => 
            _timeSeriesPredictionFunction.CheckPoint(_ml, ModelPath);
    }
}