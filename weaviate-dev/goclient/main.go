package main

import (
	"context"
	"fmt"
	"time"

	"github.com/golang/glog"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate/entities/models"
)

const (
	WeaviateHTTPEndpoint = "http://localhost:31467"
	OllamaEndpoint       = "http://localhost:30733"
)

func initWeaviateClient() (*weaviate.Client, error) {
	cfg := weaviate.Config{
		Host:       WeaviateHTTPEndpoint,
		Scheme:     "http",
		Headers:    nil,
		AuthConfig: nil,
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create weaviate driver: %v", err)
	}

	_, err = client.Misc().LiveChecker().Do(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to verify weaviate connectivity: %v", err)
	}
	glog.Info("Weaviate connection established.")
	return client, nil
}

func initCollectionSchema(client *weaviate.Client) error {
	// Define class schema
	dropletClass := &models.Class{
		Class: "Droplet",
		Properties: []*models.Property{
			{
				Name:     "uuid",
				DataType: []string{"text"},
			},
			{
				Name:     "createdTime",
				DataType: []string{"date"},
			},
			{
				Name:     "modifiedTime",
				DataType: []string{"date"},
			},
			{
				Name:     "geoCoordinates",
				DataType: []string{"geoCoordinates"},
			},
			{
				Name:     "text",
				DataType: []string{"text"},
			},
			{
				Name:     "text_format",
				DataType: []string{"text"},
			},
		},
		Vectorizer: "text2vec-ollama",
		ModuleConfig: map[string]interface{}{
			"text2vec-ollama": map[string]interface{}{
				"model":              "nomic-embed-text:latest",
				"vectorizeClassName": true,
				"baseURL":            OllamaEndpoint,
			},
			"generative-ollama": map[string]interface{}{
				"model":   "llama3.2:3b",
				"enabled": true,
				"baseURL": OllamaEndpoint,
			},
		},
		Description: "A collection representing droplets with metadata and location.",
	}

	// Create schema
	ctx := context.Background()
	err := client.Schema().ClassCreator().WithClass(dropletClass).Do(ctx)
	if err != nil {
		return fmt.Errorf("Failed to create class: %v", err)
	}
	fmt.Println("Class 'Droplet' created successfully.")
	return nil
}

func insertData(client *weaviate.Client) error {
	// Insert example Droplet object
	droplet := &models.Object{
		Class: "Droplet",
		Properties: map[string]interface{}{
			"uuid":         "abc123",
			"createdTime":  time.Now().Format(time.RFC3339),
			"modifiedTime": time.Now().Format(time.RFC3339),
			"coordinates": map[string]float64{
				"latitude":  37.7749,
				"longitude": -122.4194,
			},
			"text":        "A sample droplet of thought.",
			"text_format": "markdown",
		},
	}

	ctx := context.Background()
	_, err := client.Data().Creator().WithObject(droplet).Do(ctx)
	if err != nil {
		return fmt.Errorf("Failed to insert Droplet object: %v", err)
	}

	fmt.Println("Example Droplet object inserted successfully.")
	return nil
}

func main() {
	client, err := initWeaviateClient()
	if err != nil {
		panic(err)
	} else {
		fmt.Println("Weaviate client initialized.")
	}
	initCollectionSchema(client)
}
