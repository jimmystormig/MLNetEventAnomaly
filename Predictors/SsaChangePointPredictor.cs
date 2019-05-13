using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.TimeSeries;

namespace MLNetEventAnomaly.Predictors
{
    public class SsaChangePointPredictor
    {
        private const string ModelPath = "data/SsaChangePointPredictorModel.zip";
        private readonly MLContext _ml;
        private TimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction> _timeSeriesPredictionFunction;
    
        public SsaChangePointPredictor()
        {
            _ml = new MLContext(seed: 1, conc: 1);
        }
    
        public void Initialize()
        {
            try
            {
                using (var file = File.OpenRead(ModelPath))
                if (file != null && file.Length > 0)
                {
                    var model = TransformerChain.LoadFrom(_ml, file);
                    _timeSeriesPredictionFunction = model.CreateTimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction>(_ml);
                    return;
                }
            }
            catch (FileNotFoundException) { }
        
            const int changeHistorySize = 10;
            const int seasonalitySize = 10;
            const int numberOfSeasonsInTraining = 5;
            const int maxTrainingSize = numberOfSeasonsInTraining * seasonalitySize;

            var data = File
                .ReadAllLines("training_data/maindoorstates.csv")
                .Select(row => row.Split(';'))
                .Select(row => new { Date = DateTime.Parse(row[0], null, DateTimeStyles.AssumeUniversal), Value = int.Parse(row[1]) })
                .Select(row => new SsaChangePointPredictorData((int) row.Date.DayOfWeek, row.Date.Hour, row.Value))
                .ToList();

            _timeSeriesPredictionFunction = new SsaChangePointEstimator(_ml, new SsaChangePointDetector.Arguments()
                {
                    Confidence = 95,
                    Source = nameof(SsaChangePointPredictorData.DoorOpenings),
                    Name = nameof(SsaChangePointPredictorPrediction.Change),
                    ChangeHistoryLength = changeHistorySize,
                    TrainingWindowSize = maxTrainingSize,
                    SeasonalWindowSize = seasonalitySize
                })
                .Fit(_ml.CreateStreamingDataView(data))
                .CreateTimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction>(_ml);
        }

        public SsaChangePointPredictorPrediction Predict(DateTime timestamp, int doorOpenings) => 
            _timeSeriesPredictionFunction.Predict(new SsaChangePointPredictorData((int)timestamp.DayOfWeek, timestamp.Hour, doorOpenings));

        public void SaveModel() => 
            _timeSeriesPredictionFunction.CheckPoint(_ml, ModelPath);
    }
}