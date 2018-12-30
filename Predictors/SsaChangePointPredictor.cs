using System;
using System.Collections.Generic;
using System.IO;
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

            var pipeline = new SsaChangePointEstimator(_ml, new SsaChangePointDetector.Arguments()
            {
                Confidence = 95,
                Source = "MinutesSinceMidnight",
                Name = "Change",
                ChangeHistoryLength = changeHistorySize,
                TrainingWindowSize = maxTrainingSize,
                SeasonalWindowSize = seasonalitySize
            });

            var data = new List<SsaChangePointPredictorData>();

            for (int j = 0; j < numberOfSeasonsInTraining; j++)
            for (int i = 0; i < seasonalitySize; i++)
                data.Add(new SsaChangePointPredictorData(((i + 1) % 7) - 1, 420));

            _timeSeriesPredictionFunction = pipeline
                .Fit(_ml.CreateStreamingDataView(data))
                .CreateTimeSeriesPredictionFunction<SsaChangePointPredictorData, SsaChangePointPredictorPrediction>(_ml);
        }

        public SsaChangePointPredictorPrediction Predict(DateTime timestamp) => 
            _timeSeriesPredictionFunction.Predict(new SsaChangePointPredictorData((int)timestamp.DayOfWeek, (float)timestamp.TimeOfDay.TotalMinutes));

        public void SaveModel() => 
            _timeSeriesPredictionFunction.CheckPoint(_ml, ModelPath);
    }
}