package com.flippingai;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.*;

@Slf4j
public class ApiClient
{
    private final Gson gson = new Gson();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ConcurrentHashMap<String, CachedResponse> cache = new ConcurrentHashMap<>();

    private static final int TIMEOUT_MS = 5000;
    private static final long CACHE_TTL_MS = 10_000; // 10 seconds

    private String baseUrl;

    public ApiClient(String baseUrl)
    {
        this.baseUrl = baseUrl;
    }

    public void setBaseUrl(String baseUrl)
    {
        this.baseUrl = baseUrl;
    }

    /**
     * Get prediction for an item at a specific horizon.
     * Returns null if the request fails or times out.
     */
    public CompletableFuture<ItemPrediction> getPrediction(int itemId, String horizon)
    {
        String cacheKey = itemId + ":" + horizon;
        CachedResponse cached = cache.get(cacheKey);
        if (cached != null && !cached.isExpired())
        {
            return CompletableFuture.completedFuture(cached.prediction);
        }

        return CompletableFuture.supplyAsync(() -> {
            try
            {
                String url = baseUrl + "/api/predict/" + itemId + "?horizon=" + horizon;
                String json = httpGet(url);
                if (json == null)
                {
                    return null;
                }

                ItemPrediction prediction = parsePrediction(json);
                cache.put(cacheKey, new CachedResponse(prediction));
                return prediction;
            }
            catch (Exception e)
            {
                log.debug("Failed to get prediction for item {}: {}", itemId, e.getMessage());
                return null;
            }
        }, executor);
    }

    /**
     * Get top opportunities from the backend.
     */
    public CompletableFuture<OpportunityList> getOpportunities(int limit)
    {
        return CompletableFuture.supplyAsync(() -> {
            try
            {
                String url = baseUrl + "/api/opportunities?limit=" + limit + "&sort_by=potential_profit&sort_dir=desc";
                String json = httpGet(url);
                if (json == null)
                {
                    return null;
                }
                return gson.fromJson(json, OpportunityList.class);
            }
            catch (Exception e)
            {
                log.debug("Failed to get opportunities: {}", e.getMessage());
                return null;
            }
        }, executor);
    }

    private String httpGet(String urlStr)
    {
        try
        {
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(TIMEOUT_MS);
            conn.setRequestProperty("Accept", "application/json");
            conn.setRequestProperty("User-Agent", "FlippingAI-RuneLite/1.0");

            int status = conn.getResponseCode();
            if (status != 200)
            {
                log.debug("API returned status {}", status);
                return null;
            }

            BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null)
            {
                sb.append(line);
            }
            reader.close();
            conn.disconnect();
            return sb.toString();
        }
        catch (Exception e)
        {
            log.debug("HTTP GET failed: {}", e.getMessage());
            return null;
        }
    }

    private ItemPrediction parsePrediction(String json)
    {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        ItemPrediction pred = new ItemPrediction();
        pred.itemId = root.has("item_id") ? root.get("item_id").getAsInt() : 0;
        pred.itemName = root.has("item_name") ? root.get("item_name").getAsString() : "Unknown";
        pred.currentBuy = root.has("current_buy") ? root.get("current_buy").getAsLong() : 0;
        pred.currentSell = root.has("current_sell") ? root.get("current_sell").getAsLong() : 0;

        if (root.has("suggested_action"))
        {
            JsonObject action = root.getAsJsonObject("suggested_action");
            pred.suggestedBuy = action.has("buy_at") ? action.get("buy_at").getAsLong() : 0;
            pred.suggestedSell = action.has("sell_at") ? action.get("sell_at").getAsLong() : 0;
            pred.expectedProfit = action.has("expected_profit") ? action.get("expected_profit").getAsLong() : 0;
            pred.confidence = action.has("confidence") ? action.get("confidence").getAsDouble() : 0;
            pred.horizon = action.has("horizon") ? action.get("horizon").getAsString() : "";
        }

        if (root.has("predictions"))
        {
            JsonObject predictions = root.getAsJsonObject("predictions");
            for (String key : predictions.keySet())
            {
                JsonObject h = predictions.getAsJsonObject(key);
                if (h.has("direction"))
                {
                    pred.directions.put(key, h.get("direction").getAsString());
                }
                if (h.has("confidence"))
                {
                    pred.confidences.put(key, h.get("confidence").getAsDouble());
                }
            }
        }

        return pred;
    }

    public void shutdown()
    {
        executor.shutdown();
    }

    private static class CachedResponse
    {
        final ItemPrediction prediction;
        final long timestamp;

        CachedResponse(ItemPrediction prediction)
        {
            this.prediction = prediction;
            this.timestamp = System.currentTimeMillis();
        }

        boolean isExpired()
        {
            return System.currentTimeMillis() - timestamp > CACHE_TTL_MS;
        }
    }
}
