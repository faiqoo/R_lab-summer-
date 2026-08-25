<?php

namespace Drupal\chatbot\Plugin\QueueWorker;

use Drupal\Core\Entity\EntityTypeManagerInterface;
use Drupal\Core\Plugin\ContainerFactoryPluginInterface;
use Drupal\Core\Queue\QueueWorkerBase;
use GuzzleHttp\ClientInterface;
use GuzzleHttp\Exception\RequestException;
use Symfony\Component\DependencyInjection\ContainerInterface;

/**
 * Uploads newly added documents to the RAG backend.
 *
 * @QueueWorker(
 *   id = "chatbot_document_ingest",
 *   title = @Translation("Chatbot Document Ingest"),
 *   cron = {"time" = 30}
 * )
 */
class DocumentIngestQueue extends QueueWorkerBase implements ContainerFactoryPluginInterface {

  /**
   * Entity type manager.
   *
   * @var \Drupal\Core\Entity\EntityTypeManagerInterface
   */
  protected $entityTypeManager;

  /**
   * HTTP client.
   *
   * @var \GuzzleHttp\ClientInterface
   */
  protected $httpClient;

  /**
   * Constructor.
   */
  public function __construct(
    array $configuration,
    $plugin_id,
    $plugin_definition,
    EntityTypeManagerInterface $entityTypeManager,
    ClientInterface $httpClient
  ) {
    parent::__construct($configuration, $plugin_id, $plugin_definition);

    $this->entityTypeManager = $entityTypeManager;
    $this->httpClient = $httpClient;
  }

  /**
   * {@inheritdoc}
   */
  public static function create(
    ContainerInterface $container,
    array $configuration,
    $plugin_id,
    $plugin_definition
  ) {
    return new static(
      $configuration,
      $plugin_id,
      $plugin_definition,
      $container->get('entity_type.manager'),
      $container->get('http_client')
    );
  }

  /**
   * {@inheritdoc}
   */
  public function processItem($data) {

    if (empty($data['media_id'])) {
      return;
    }

    $media = $this->entityTypeManager
      ->getStorage('media')
      ->load($data['media_id']);

    if (!$media) {
      \Drupal::logger('chatbot')->warning(
        'Media @id no longer exists.',
        ['@id' => $data['media_id']]
      );
      return;
    }

    if (!$media->hasField('field_media_document')) {
      \Drupal::logger('chatbot')->warning(
        'Media @id has no field_media_document.',
        ['@id' => $media->id()]
      );
      return;
    }

    $file = $media->get('field_media_document')->entity;

    if (!$file) {
      \Drupal::logger('chatbot')->warning(
        'No file attached to media @id.',
        ['@id' => $media->id()]
      );
      return;
    }

    $uri = $file->getFileUri();

    $stream = fopen($uri, 'rb');

    if (!$stream) {
      \Drupal::logger('chatbot')->error(
        'Unable to open stream: @uri',
        ['@uri' => $uri]
      );
      return;
    }

    try {

      $response = $this->httpClient->post(
        'http://host.docker.internal:8000/ingest',
        [
          'headers' => [
            'X-Api-Key' => '4f5d40713235afe302297da1c9c310d5',
          ],
          'multipart' => [
            [
              'name' => 'files',
              'contents' => $stream,
              'filename' => $file->getFilename(),
            ],
          ],
          'timeout' => 120,
        ]
      );

      \Drupal::logger('chatbot')->notice(
        'Successfully ingested "@file". Status: @status',
        [
          '@file' => $file->getFilename(),
          '@status' => $response->getStatusCode(),
        ]
      );

      \Drupal::logger('chatbot')->notice(
        'Backend response: @response',
        [
          '@response' => (string) $response->getBody(),
        ]
      );

    }
    catch (RequestException $e) {

      \Drupal::logger('chatbot')->error(
        'Ingestion failed for "@file": @message',
        [
          '@file' => $file->getFilename(),
          '@message' => $e->getMessage(),
        ]
      );

      if ($e->hasResponse()) {
        \Drupal::logger('chatbot')->error(
          'Backend response: @response',
          [
            '@response' => (string) $e->getResponse()->getBody(),
          ]
        );
      }

    }
finally {

  if (is_resource($stream)) {
    fclose($stream);
  }

}

    }
    
    }