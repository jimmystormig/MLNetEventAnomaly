using System;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
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
            var predictor = new Predictor();
            predictor.Initialize();

            applicationLifetime.ApplicationStopping.Register(() =>
            {
                predictor.SaveModel();
            });

            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            
            app.Run(async (context) =>
            {
                switch (context.Request)
                {
                    case HttpRequest predictPost when 
                        predictPost.Path.Equals("/predict") && 
                        predictPost.Method.Equals("POST") &&
                        DateTime.TryParse(predictPost.Query["timestamp"], out var timestamp):
                        var prediction = predictor.Predict(timestamp);
                        await context.Response.WriteAsync(JsonConvert.SerializeObject(prediction));
                        break;
                    default:
                        await context.Response.WriteAsync("Hello World!");
                        break;
                }
            });
        }
    }
}
