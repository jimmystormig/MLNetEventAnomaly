using System;
using System.Globalization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using MLNetEventAnomaly;
using MLNetEventAnomaly.Predictors;
using Newtonsoft.Json;

namespace EventAnomaly
{
    public class Startup
    {
        // This method gets called by the runtime. Use this method to add services to the container.
        // For more information on how to configure your application, visit https://go.microsoft.com/fwlink/?LinkID=398940
        public void ConfigureServices(IServiceCollection services)
        {
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env, IApplicationLifetime applicationLifetime)
        {
            var randomizedPcaPredictor = new RandomizedPcaPredictor();
            randomizedPcaPredictor.Train();
            randomizedPcaPredictor.Initialize();
            
            var ssaChangePointPredictor = new SsaChangePointPredictor();
            ssaChangePointPredictor.Initialize();

            applicationLifetime.ApplicationStopping.Register(() =>
            {
                ssaChangePointPredictor.SaveModel();
            });
             
            app.Run(async (context) =>
            {
                switch (context.Request)
                {
                    case HttpRequest predictPost when 
                        predictPost.Path.Equals("/predict") &&
                        predictPost.Method.Equals("POST") &&
                        predictPost.Query.TryGetValue("algorithm", out var algorithm) &&
                        DateTime.TryParse(predictPost.Query["timestamp"], null, DateTimeStyles.AssumeUniversal, out var timestamp):
                        switch (algorithm)
                        {
                            case "ssachangepoint":
                                var ssaChangePointPrediction = ssaChangePointPredictor.Predict(timestamp);
                                await context.Response.WriteAsync(JsonConvert.SerializeObject(ssaChangePointPrediction));
                                break;
                            case "randomizedpca":
                                var randomizedPcaPredictorPrediction = randomizedPcaPredictor.Predict(timestamp);
                                await context.Response.WriteAsync(JsonConvert.SerializeObject(randomizedPcaPredictorPrediction));
                                break;
                            default:
                                throw new ArgumentOutOfRangeException("algorithm", algorithm, "Invalid algorithm");
                        }
                        break;
                    default:
                        await context.Response.WriteAsync("It's alive!");
                        break;
                }
            });
        }
    }
}
