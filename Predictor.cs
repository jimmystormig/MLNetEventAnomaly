using System;
using System.Collections.Generic;
using System.IO;
using EventAnomaly;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.TimeSeries;

public class Predictor
{
    private const string ModelPath = "temp.zip";
    private readonly MLContext _ml;
    private TimeSeriesPredictionFunction<Data, Prediction> _timeSeriesPredictionFunction;
    
    public Predictor()
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
                    _timeSeriesPredictionFunction = model.CreateTimeSeriesPredictionFunction<Data, Prediction>(_ml);
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

        List<Data> data = new List<Data>();

        var ml = new MLContext(seed: 1, conc: 1);
        var dataView = ml.CreateStreamingDataView(data);

        for (int j = 0; j < numberOfSeasonsInTraining; j++)
        for (int i = 0; i < seasonalitySize; i++)
            data.Add(new Data(1, i));

        for (int i = 0; i < changeHistorySize; i++)
            data.Add(new Data(1, i * 100));
        
        _timeSeriesPredictionFunction = pipeline
            .Fit(_ml.CreateStreamingDataView(data))
            .CreateTimeSeriesPredictionFunction<Data, Prediction>(_ml);
    }

    public Prediction Predict(DateTime timestamp) => 
        _timeSeriesPredictionFunction.Predict(new Data((int)timestamp.DayOfWeek, (float)timestamp.TimeOfDay.TotalMinutes));

    public void SaveModel() => 
        _timeSeriesPredictionFunction.CheckPoint(_ml, ModelPath);
}